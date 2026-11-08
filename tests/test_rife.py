"""
RIFE VFI (Unica)'s loop (comfyui/rife.py) against the original loop's order, without ComfyUI: a stand-in model with
IFNet's signature blends the two frames at the timestep, so every output frame says where it came from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfyui.rife import interpolate  # noqa: E402

SCALES = [8, 4, 2, 1]


def blend(img0, img1, timestep, scale_list, training, fastmode):
    """IFNet's positional signature; the timestep comes as a [batch, 1, 1, 1] tensor."""
    assert scale_list == SCALES and training is True and fastmode is False
    assert img0.shape == img1.shape and img0.shape[0] == timestep.shape[0] and img0.shape[1] == 3
    return img0 * (1 - timestep) + img1 * timestep


def test_interpolate_keeps_the_original_order():
    torch.manual_seed(0)
    frames = torch.rand(4, 6, 10, 4)  # [n, h, w, c]; a 4th channel, which the node drops
    multiplier = 5
    out = interpolate(blend, frames, multiplier, SCALES, True, False, torch.device("cpu"))

    rgb = frames[..., :3]
    expected = []
    for index in range(len(frames) - 1):  # the original: each frame, then the middle frames one at a time
        expected.append(rgb[index])
        for step in range(1, multiplier):
            t = torch.tensor(step / multiplier)
            expected.append(rgb[index] * (1 - t) + rgb[index + 1] * t)
    expected.append(rgb[-1])
    assert out.shape == (len(expected), 6, 10, 3) and out.dtype == torch.float32 and out.is_contiguous()
    assert torch.allclose(out, torch.stack(expected), atol=1e-6)


def test_multiplier_one_returns_the_frames():
    frames = torch.rand(3, 4, 4, 3)
    assert torch.equal(interpolate(blend, frames, 1, SCALES, True, False, torch.device("cpu")), frames)
