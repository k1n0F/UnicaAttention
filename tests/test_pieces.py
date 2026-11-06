"""M4.31: a call split by heads, to keep its temporary buffers in the VRAM, and float16 input converted piece by
piece, give bit for bit what one bfloat16 call gives."""
import pytest
import torch

import unica

SHAPE = (1, 8, 700, 128)  # 700 tokens: a partial tile at the end


@pytest.fixture
def qkv():
    torch.manual_seed(0)
    return [torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16) for _ in range(3)]


@pytest.mark.parametrize("preset", list(unica.PRESETS))
def test_pieces_are_one_call(qkv, preset, monkeypatch):
    settings = unica.PRESETS[preset]
    whole = unica.attention(*qkv, **settings)
    monkeypatch.setattr(unica, "heads_per_call", lambda *args: 3)  # pieces of 3, 3 and 2 heads
    assert torch.equal(unica.attention(*qkv, **settings), whole)
    half = unica.attention(*(x.half() for x in qkv), **settings)  # converted per piece, written as fp16
    assert half.dtype == torch.float16 and torch.equal(half, whole.half())


def test_pieces_keep_the_callers_layout(qkv, monkeypatch):
    monkeypatch.setattr(unica, "heads_per_call", lambda *args: 3)
    nhd = [x.transpose(1, 2).contiguous().half() for x in qkv]  # ComfyUI's [batch, tokens, heads, 128] in fp16
    out = unica.attention(*(x.transpose(1, 2) for x in nhd))
    assert out.transpose(1, 2).is_contiguous()  # written in place in the caller's layout
    assert torch.equal(out, unica.attention(*qkv).half())


def test_heads_per_call_follows_the_free_vram(qkv, monkeypatch):
    q, k, _ = qkv
    per_head = 1 * 128 * 3 * SHAPE[2] * 2  # bf16 input: 2 bytes per element of q, k, v
    monkeypatch.setattr(unica, "VRAM_RESERVE", 512 * 2**20)
    monkeypatch.setattr(unica, "free_vram", lambda device: unica.VRAM_RESERVE + 5 * per_head)
    assert unica.heads_per_call(q, k, unica.DEFAULT_KERNEL, {}) == 5
    assert unica.heads_per_call(q.half(), k.half(), unica.DEFAULT_KERNEL, {}) == 2  # fp16 needs its bf16 copies too
    monkeypatch.setattr(unica, "free_vram", lambda device: 0)
    assert unica.heads_per_call(q, k, unica.DEFAULT_KERNEL, {}) == 1  # never below one head


def test_the_automatic_reserve_is_a_tenth_of_the_card_or_1_gib(monkeypatch):
    monkeypatch.setattr(unica, "VRAM_RESERVE", None)
    total = torch.cuda.get_device_properties(0).total_memory
    assert unica.vram_reserve(torch.device("cuda")) == max(2**30, total // 10)
