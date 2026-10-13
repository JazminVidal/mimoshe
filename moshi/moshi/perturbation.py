"""Live periodic perturbations of temporal-transformer residual activations."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass
class PerturbationParams:
    enabled: bool = False
    layer: int = 1  # One-based layer number, applied after that block.
    strength: float = 0.2  # Added vector RMS / current activation RMS.
    frequency: float = 0.25
    harmonics: int = 3
    seed: int = 1234
    preset: str = "orbit"
    walk_step: float = 0.1  # RMS of each random-walk increment, before strength.


class CyclicPerturbation:
    """Add a seeded cyclic trajectory at one temporal transformer layer.

    Hooks are installed before CUDA graph capture. Every hook always reads its
    fixed-address GPU buffer, including when disabled; prepare_step updates the
    buffers outside graph replay. Live changes therefore do not recapture graphs
    or reset KV caches. The same vector is broadcast over the batch (also CFG).
    """

    def __init__(self, transformer: nn.Module, dimension: int, frame_rate: float):
        layers = transformer.layers
        parameter = next(transformer.parameters())
        self.layer_count = len(layers)
        self.dimension = dimension
        self.frame_rate = frame_rate
        self.max_harmonics = min(6, dimension // 2)
        # Allow fast experimental motion; higher harmonics can intentionally
        # alias at the model's token-frame rate.
        self.max_frequency = 6.0
        self.params = PerturbationParams(
            layer=max(1, self.layer_count // 2),
            harmonics=min(3, self.max_harmonics),
        )
        self.offsets = torch.zeros(
            self.layer_count, dimension,
            device=parameter.device, dtype=parameter.dtype,
        )
        self.phase = 0.0
        self._cycle_start = True
        self._walk_index = 0
        self.smoothed_strength = 0.0
        self._basis = self._make_basis()
        self._pulse_vector = self._make_pulse_vector()
        self._walk_displacements = None
        self._handles = []
        for index, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(index)))

    def _make_hook(self, index: int):
        def hook(_module, _args, output):
            # Float32 accumulation avoids overflow in half precision. This is
            # entirely tensor code and remains dynamic under CUDA graph replay.
            rms = output.float().square().mean(dim=-1, keepdim=True).sqrt()
            return output + (rms * self.offsets[index]).to(output.dtype)
        return hook

    def _make_basis(self):
        generator = torch.Generator(device="cpu").manual_seed(self.params.seed)
        random_vectors = torch.randn(
            self.dimension, self.params.harmonics * 2, generator=generator,
        )
        # Orthonormal directions make the orbit RMS constant at every phase.
        basis, _ = torch.linalg.qr(random_vectors, mode="reduced")
        return basis * math.sqrt(self.dimension)

    def trajectory(self, phase: float):
        """The original smooth Fourier orbit, not independent random noise."""
        coefficients = []
        for harmonic in range(1, self.params.harmonics + 1):
            angle = math.tau * harmonic * phase
            coefficients.extend([math.cos(angle) / harmonic, math.sin(angle) / harmonic])
        weights = torch.tensor(coefficients, dtype=torch.float32)
        weights /= weights.square().sum().sqrt()
        return self._basis @ weights

    def _make_pulse_vector(self):
        generator = torch.Generator(device="cpu").manual_seed(self.params.seed)
        vector = torch.randn(self.dimension, generator=generator)
        return vector / vector.square().mean().sqrt()

    def _make_walk(self):
        # Cache enough directions for the longest supported cycle (0.01 Hz).
        # This private generator never consumes Moshi's sampling RNG.
        count = math.ceil(self.frame_rate / 0.01) + 2
        generator = torch.Generator(device="cpu").manual_seed(self.params.seed ^ 0x573A72)
        directions = torch.randn(count, self.dimension, generator=generator)
        directions /= directions.square().mean(dim=-1, keepdim=True).sqrt()
        return torch.cat([
            torch.zeros(1, self.dimension), directions.cumsum(dim=0),
        ])

    def update(self, data: dict):
        previous_basis_key = (self.params.seed, self.params.harmonics)
        previous_preset = self.params.preset
        limits = {
            "layer": (int, 1, self.layer_count),
            "strength": (float, 0.0, 10.0),
            "frequency": (float, 0.01, self.max_frequency),
            "harmonics": (int, 1, self.max_harmonics),
            "seed": (int, 0, 2**31 - 1),
            "walk_step": (float, 0.0, 1.0),
        }
        if isinstance(data.get("enabled"), bool):
            self.params.enabled = data["enabled"]
        if data.get("preset") in ("orbit", "pulse", "walk"):
            self.params.preset = data["preset"]
        for name, (cast, minimum, maximum) in limits.items():
            if name not in data or isinstance(data[name], bool):
                continue
            try:
                value = cast(data[name])
                if not math.isfinite(value):
                    continue
            except (ValueError, TypeError, OverflowError):
                continue
            setattr(self.params, name, max(minimum, min(maximum, value)))
        if previous_basis_key != (self.params.seed, self.params.harmonics):
            self._basis = self._make_basis()
        if previous_basis_key[0] != self.params.seed:
            self._pulse_vector = self._make_pulse_vector()
            self._walk_displacements = None
        if self.params.preset == "walk":
            if self._walk_displacements is None:
                self._walk_displacements = self._make_walk()
            if previous_preset != "walk":
                # Enter the path at the current cycle position, not phase zero.
                self._walk_index = int(self.phase * self.frame_rate / self.params.frequency)
        if not self.params.enabled or self.params.strength == 0:
            self.offsets.zero_()
            self.smoothed_strength = 0.0
        return self.status()

    @torch.no_grad()
    def prepare_step(self):
        self.offsets.zero_()
        if self.params.enabled and self.params.strength > 0:
            smoothing = 1 - math.exp(-1 / (self.frame_rate * 0.15))
            self.smoothed_strength += smoothing * (
                self.params.strength - self.smoothed_strength
            )
            if self.params.preset == "pulse":
                # Exactly one inference frame at each cycle start. No smoothing
                # tail: every other frame's injection is exactly zero.
                if self._cycle_start:
                    self.offsets[self.params.layer - 1].copy_(
                        self._pulse_vector * self.params.strength,
                    )
            else:
                if self.params.preset == "walk":
                    assert self._walk_displacements is not None
                    if self._cycle_start:
                        self._walk_index = 0
                    index = min(self._walk_index, len(self._walk_displacements) - 1)
                    vector = self._pulse_vector + self.params.walk_step * self._walk_displacements[index]
                    self._walk_index += 1
                else:
                    vector = self.trajectory(self.phase)
                self.offsets[self.params.layer - 1].copy_(vector * self.smoothed_strength)
        else:
            self.smoothed_strength = 0.0
            if self.params.preset == "walk":
                if self._cycle_start:
                    self._walk_index = 0
                self._walk_index += 1
        next_phase = self.phase + self.params.frequency / self.frame_rate
        self._cycle_start = next_phase >= 1.0
        self.phase = next_phase % 1.0

    def reset_phase(self):
        self.phase = 0.0
        self._cycle_start = True
        self._walk_index = 0
        self.smoothed_strength = 0.0
        self.offsets.zero_()

    def status(self):
        return {
            "params": asdict(self.params),
            "layer_count": self.layer_count,
            "dimension": self.dimension,
            "frame_rate": self.frame_rate,
            "max_harmonics": self.max_harmonics,
            "max_frequency": self.max_frequency,
            "phase": self.phase,
        }

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
