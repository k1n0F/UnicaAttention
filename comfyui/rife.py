"""
RIFE VFI (Unica): the RIFE node of comfyui-frame-interpolation with its loop rebuilt (LOG M4.37). The model, the
checkpoints and their download stay that package's; only the loop around the model is new.

The original loop, per pair of frames: the middle frames one at a time, both frames uploaded again for each, each
result copied back with a sync, then gc.collect() (~0.67 s in ComfyUI's heap, 112 calls in "wan stable work"), into
a zero-filled float32 output in [n, 3, h, w] that ComfyUI then reads through a permuted view. In one render that was
140 s, 56 % of it in gc.collect and the GPU busy 21 % of the time.

Here: each frame is uploaded once, a pair's middle frames run as one batch, the output is allocated without the fill
and written in ComfyUI's [n, h, w, 3] layout, and there is no gc.collect. The model gets the same arguments in the same
positions as in the original (its fast_mode lands in IFNet's `training`, its ensemble in `fastmode`), so the frames
equal the original's up to the rounding of batched convolutions.
"""

# pyright: reportMissingImports=false
# ComfyUI's modules (comfy, cv2, the frame-interpolation node...) exist only in ComfyUI's Python, which runs this file

from __future__ import annotations

import torch


def interpolate(model, frames: torch.Tensor, multiplier: int, scales: list[float], fast_mode: bool, ensemble: bool,
                device: torch.device) -> torch.Tensor:
    """frames: ComfyUI's IMAGE, float32 [n, h, w, c] on the CPU. Returns [(n - 1) · multiplier + 1, h, w, 3]: each input
    frame followed by multiplier - 1 middle frames at timesteps 1/multiplier … (multiplier - 1)/multiplier."""
    count, height, width, _ = frames.shape
    steps = multiplier - 1
    out = torch.empty((count - 1) * multiplier + 1, height, width, 3)
    timesteps = (torch.arange(1, multiplier, device=device, dtype=torch.float32) / multiplier).view(-1, 1, 1, 1)

    def upload(index: int) -> torch.Tensor:
        return frames[index, ..., :3].to(device).permute(2, 0, 1)[None]

    frame_1 = upload(0)
    for index in range(count - 1):
        frame_0, frame_1 = frame_1, upload(index + 1)
        first = index * multiplier
        out[first] = frames[index, ..., :3]
        if steps:
            middle = model(frame_0.expand(steps, -1, -1, -1), frame_1.expand(steps, -1, -1, -1), timesteps, scales,
                           fast_mode, ensemble)
            out[first + 1:first + multiplier] = middle.permute(0, 2, 3, 1).contiguous()
    out[-1] = frames[-1, ..., :3]
    return out


def checkpoint_names() -> list[str]:
    try:
        from vfi_models.rife import CKPT_NAME_VER_DICT  # on sys.path once comfyui-frame-interpolation has loaded
    except ImportError:
        return ["rife49.pth"]
    return list(CKPT_NAME_VER_DICT)


class UnicaRIFE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (checkpoint_names(), {"default": "rife49.pth"}),
                "frames": ("IMAGE",),
                "multiplier": ("INT", {"default": 2, "min": 1}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "ensemble": ("BOOLEAN", {"default": True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "vfi"
    CATEGORY = "UnicaAttention"
    DESCRIPTION = ("RIFE VFI of comfyui-frame-interpolation (its models), with a faster loop: one upload per frame, "
                   "a pair's middle frames in one batch, no gc.collect per pair. Needs comfyui-frame-interpolation.")

    @torch.inference_mode()
    def vfi(self, ckpt_name, frames, multiplier, fast_mode, ensemble, scale_factor):
        import comfy.model_management
        from vfi_models.rife import CKPT_NAME_VER_DICT, MODEL_TYPE
        from vfi_models.rife.rife_arch import IFNet
        from vfi_utils import load_file_from_github_release

        device = comfy.model_management.get_torch_device()
        model = IFNet(arch_ver=CKPT_NAME_VER_DICT[ckpt_name])
        model.load_state_dict(torch.load(load_file_from_github_release(MODEL_TYPE, ckpt_name)))
        model.eval().to(device)
        scales = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]
        out = interpolate(model, frames, multiplier, scales, fast_mode, ensemble, device)
        del model
        comfy.model_management.soft_empty_cache()
        return (out,)
