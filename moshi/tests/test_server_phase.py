import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import aiohttp
import numpy as np
import torch

from moshi.server import ServerState


class FakeSocket:
    def __init__(self, payloads):
        self.messages = iter(payloads)
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=next(self.messages))
        except StopIteration:
            raise StopAsyncIteration

    async def send_bytes(self, data):
        self.sent.append(data)


def make_state():
    # Exercise the real receiving loop without loading any model weights.
    state = object.__new__(ServerState)
    state.device = "cpu"
    state.frame_size = 4
    state.mimi = SimpleNamespace(
        encode=lambda chunk: torch.zeros(1, 1, 1, dtype=torch.long),
        reset_streaming=Mock(),
    )
    state.perturbation = SimpleNamespace(phase=0.0)

    def prepare_step():
        state.perturbation.phase = (state.perturbation.phase + 0.125) % 1

    def reset_phase():
        state.perturbation.phase = 0

    state.perturbation.prepare_step = prepare_step
    state.perturbation.reset_phase = reset_phase
    state.lm_gen = SimpleNamespace(
        step=Mock(side_effect=[None] + [torch.zeros(1, 2, 1)] * 10),
        reset_streaming=Mock(),
    )

    async def decode_and_send(tokens, ws, writer):
        await ws.send_bytes(b"\x01audio")

    state.decode_and_send = decode_and_send
    return state


def test_phase_message_for_every_step_even_batched_and_startup_audio():
    state = make_state()
    ws = FakeSocket([b"\x01four-frames", b"\x01no-pcm", b"\x01two-frames"])
    reader = SimpleNamespace(append_bytes=Mock(side_effect=[
        np.zeros(16, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros(8, dtype=np.float32),
    ]))
    asyncio.run(state.recv_loop(ws, reader, None))
    phase_messages = [json.loads(data[1:]) for data in ws.sent if data[0] == 4]
    assert phase_messages == [
        {"perturbation_phase": step * 0.125} for step in range(1, 7)
    ]
    assert state.lm_gen.step.call_count == 6
    assert ws.sent[0][0] == 4  # startup step reports despite tokens=None
    assert sum(data[0] == 1 for data in ws.sent) == 5


def test_context_reset_reports_phase_zero_immediately():
    state = make_state()
    state.perturbation.phase = 0.75
    ws = FakeSocket([b"\x03\x03"])
    asyncio.run(state.recv_loop(ws, None, None))
    assert json.loads(ws.sent[0][1:]) == {
        "stream_reset": True, "reason": "user", "perturbation_phase": 0,
    }
