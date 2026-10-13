import math

import pytest
import torch
from torch import nn

from moshi.perturbation import CyclicPerturbation
from moshi.models.lm import LMModel
from moshi.utils.compile import CUDAGraphed


class ToyTransformer(nn.Module):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(32, 32, bias=False, device=device, dtype=dtype),
            nn.Linear(32, 32, bias=False, device=device, dtype=dtype),
        ])
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(torch.eye(32, device=device, dtype=dtype))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_closed_orbit_constant_rms_and_seed_reproducibility():
    orbit = CyclicPerturbation(ToyTransformer(), 32, 12.5)
    first = orbit.trajectory(0.13)
    torch.testing.assert_close(first, orbit.trajectory(1.13))
    for phase in (0, 0.13, 0.5, 0.999):
        assert orbit.trajectory(phase).square().mean().sqrt().item() == pytest.approx(1.0)
    orbit.update({"seed": 2345})
    assert not torch.allclose(first, orbit.trajectory(0.13))
    orbit.update({"seed": 1234})
    torch.testing.assert_close(first, orbit.trajectory(0.13))


@torch.no_grad()
def test_bypass_strength_layer_changes_and_phase_continuity():
    transformer = ToyTransformer()
    orbit = CyclicPerturbation(transformer, 32, 12.5)
    x = torch.ones(1, 1, 32)
    assert torch.equal(transformer(x), x)
    orbit.update({"enabled": True, "strength": 0.4, "layer": 1})
    orbit.prepare_step()
    y = transformer(x)
    torch.testing.assert_close(y - x, orbit.offsets[0].view_as(x))
    assert not torch.equal(y, x)
    assert torch.count_nonzero(orbit.offsets[1]) == 0
    previous_phase = orbit.phase
    orbit.update({"layer": 2, "frequency": 0.7})
    assert orbit.phase == previous_phase
    orbit.prepare_step()
    assert torch.count_nonzero(orbit.offsets[0]) == 0
    assert torch.count_nonzero(orbit.offsets[1]) > 0
    assert orbit.phase == pytest.approx((previous_phase + 0.7 / 12.5) % 1)
    orbit.update({"strength": 0})
    assert torch.equal(transformer(x), x)
    orbit.prepare_step()
    assert torch.equal(transformer(x), x)
    orbit.update({"strength": 0.4})
    orbit.prepare_step()
    assert not torch.equal(transformer(x), x)
    orbit.update({"enabled": False})
    assert torch.equal(transformer(x), x)
    orbit.reset_phase()
    assert orbit.phase == 0
    assert orbit.params.layer == 2
    assert orbit.params.frequency == 0.7


def test_invalid_controls_and_no_sampling_rng_changes():
    orbit = CyclicPerturbation(ToyTransformer(), 32, 12.5)
    state = torch.random.get_rng_state().clone()
    orbit.update({"seed": 333, "layer": 999, "frequency": math.inf, "strength": math.nan,
                  "harmonics": -1, "enabled": "true"})
    assert torch.equal(state, torch.random.get_rng_state())
    assert orbit.params.layer == 2
    assert orbit.params.frequency == 0.25
    assert orbit.params.strength == 0.2
    assert orbit.params.harmonics == 1
    assert orbit.params.enabled is False
    orbit.update({"frequency": 6, "harmonics": 6})
    assert orbit.params.frequency == 6
    assert orbit.params.harmonics == 6
    assert orbit.status()["max_frequency"] == 6
    orbit.update({"frequency": 100})
    assert orbit.params.frequency == 6
    orbit.update({"frequency": -1})
    assert orbit.params.frequency == 0.01
    orbit.update({"preset": "unknown", "walk_step": math.nan})
    assert orbit.params.preset == "orbit"
    assert orbit.params.walk_step == 0.1
    orbit.update({"strength": 10})
    assert orbit.params.strength == 10
    orbit.update({"strength": 100})
    assert orbit.params.strength == 10
    orbit.update({"strength": -1})
    assert orbit.params.strength == 0


@torch.no_grad()
def test_cycle_start_pulse_has_exact_zero_between_kicks():
    transformer = ToyTransformer()
    orbit = CyclicPerturbation(transformer, 32, 8)
    orbit.update({"preset": "pulse", "enabled": True, "frequency": 2, "strength": 0.4})
    x = torch.ones(1, 1, 32)
    for step in range(12):
        orbit.prepare_step()
        if step % 4 == 0:
            assert orbit.offsets[orbit.params.layer - 1].square().mean().sqrt().item() == pytest.approx(0.4)
            assert not torch.equal(transformer(x), x)
        else:
            assert torch.count_nonzero(orbit.offsets) == 0
            assert torch.equal(transformer(x), x)
    orbit.reset_phase()
    orbit.prepare_step()
    torch.testing.assert_close(orbit.offsets[orbit.params.layer - 1], orbit._pulse_vector * 0.4)
    orbit.update({"strength": 10})
    orbit.reset_phase()
    orbit.prepare_step()
    assert torch.isfinite(transformer(x)).all()
    assert orbit.offsets[orbit.params.layer - 1].square().mean().sqrt().item() == pytest.approx(10)
    orbit.update({"enabled": False})
    orbit.prepare_step()
    assert torch.equal(transformer(x), x)


@torch.no_grad()
def test_walk_step_amplitude_repeats_and_jumps_back_at_boundary():
    orbit = CyclicPerturbation(ToyTransformer(), 32, 8)
    random_state = torch.random.get_rng_state().clone()
    orbit.update({"preset": "walk", "enabled": True, "frequency": 2, "walk_step": 0.2})
    assert torch.equal(random_state, torch.random.get_rng_state())
    vectors = []
    for _ in range(9):
        orbit.prepare_step()
        vectors.append(orbit.offsets[orbit.params.layer - 1].clone() / orbit.smoothed_strength)
    for index in range(4):
        torch.testing.assert_close(vectors[index], vectors[index + 4])
    torch.testing.assert_close(vectors[0], vectors[8])
    for index in range(1, 4):
        increment_rms = (vectors[index] - vectors[index - 1]).square().mean().sqrt().item()
        assert increment_rms == pytest.approx(0.2)
    assert not torch.allclose(vectors[3], vectors[4])  # intentional reset jump
    phase, index = orbit.phase, orbit._walk_index
    orbit.update({"frequency": 1, "walk_step": 0.3, "seed": 5432})
    assert orbit.phase == phase and orbit._walk_index == index
    assert torch.equal(random_state, torch.random.get_rng_state())
    orbit.update({"seed": 1234, "walk_step": 0.2, "frequency": 2})
    orbit.reset_phase()
    orbit.prepare_step()
    torch.testing.assert_close(orbit.offsets[orbit.params.layer - 1] / orbit.smoothed_strength, vectors[0])


def test_preset_switch_does_not_reset_phase_and_walk_zero_holds_vector():
    orbit = CyclicPerturbation(ToyTransformer(), 32, 8)
    orbit.update({"enabled": True, "frequency": 2})
    orbit.prepare_step()
    phase = orbit.phase
    orbit.update({"preset": "walk", "walk_step": 0})
    assert orbit.phase == phase
    for _ in range(5):
        orbit.prepare_step()
        torch.testing.assert_close(
            orbit.offsets[orbit.params.layer - 1] / orbit.smoothed_strength,
            orbit._pulse_vector,
        )
    phase = orbit.phase
    orbit.update({"preset": "pulse"})
    assert orbit.phase == phase


@torch.no_grad()
@pytest.mark.parametrize("preset", ["orbit", "pulse", "walk"])
def test_real_moshi_hidden_state_affects_text_and_audio_logits(preset):
    model = LMModel(
        delays=[0, 1, 1], n_q=2, dep_q=2, card=32, text_card=48,
        dim=16, num_layers=2, num_heads=1, hidden_scale=1,
        depformer_dim=16, depformer_num_heads=1, context=4,
    ).eval()
    sequence = torch.zeros(1, 3, 1, dtype=torch.long)
    baseline_hidden, baseline_text = model.forward_text(sequence)
    baseline_audio = model.forward_depformer_training(sequence, baseline_hidden)
    orbit = CyclicPerturbation(model.transformer, 16, 12.5)
    bypass_hidden, bypass_text = model.forward_text(sequence)
    assert torch.equal(baseline_hidden, bypass_hidden)
    assert torch.equal(baseline_text, bypass_text)
    orbit.update({"enabled": True, "strength": 0.5, "preset": preset})
    orbit.prepare_step()
    hidden, text = model.forward_text(sequence)
    audio = model.forward_depformer_training(sequence, hidden)
    assert torch.isfinite(hidden).all() and torch.isfinite(audio).all()
    assert not torch.equal(text, baseline_text)
    assert not torch.equal(audio, baseline_audio)
    orbit.close()
    assert torch.equal(model.forward_text(sequence)[0], baseline_hidden)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_live_changes_reach_already_captured_cuda_graph(dtype):
    transformer = ToyTransformer("cuda", dtype)
    orbit = CyclicPerturbation(transformer, 32, 12.5)
    forward = CUDAGraphed(transformer.forward)
    x = torch.ones(1, 1, 32, device="cuda", dtype=dtype)
    forward(x)  # warmup
    baseline = forward(x).clone()  # capture while bypassed
    graph = forward._graph
    orbit.update({"enabled": True, "layer": 2, "strength": 0.5})
    orbit.prepare_step()
    modified = forward(x).clone()
    assert not torch.equal(modified, baseline)
    assert torch.isfinite(modified).all()
    orbit.update({"layer": 1, "seed": 555})
    orbit.prepare_step()
    assert not torch.equal(forward(x), modified)
    for preset in ("pulse", "walk", "orbit"):
        phase = orbit.phase
        orbit.update({"preset": preset, "frequency": 6})
        assert orbit.phase == phase
        orbit.reset_phase()
        orbit.prepare_step()
        output = forward(x).clone()
        assert torch.isfinite(output).all()
        assert not torch.equal(output, baseline)
        if preset == "pulse":
            orbit.prepare_step()
            assert torch.equal(forward(x), baseline)  # no pulse tail in graph
    orbit.update({"enabled": False})
    assert torch.equal(forward(x), baseline)
    orbit.update({"enabled": True, "strength": 10})
    for _ in range(4):
        orbit.prepare_step()
        assert torch.isfinite(forward(x)).all()
    orbit.update({"enabled": False})
    assert torch.equal(forward(x), baseline)
    assert forward._graph is graph  # no graph recapture / no model reset
