"""M4.37: the RIFE node's own loop (vfi_utils.generic_frame_loop, rife49, ×5, fast_mode) at the render's resolution,
for a given number of frames, with the memory watched, against RIFE VFI (Unica)'s loop (comfyui/rife.py) on the same
frames in the same process. Run it with ComfyUI's Python and its thread variables:

    <ComfyUI>/.venv-13/Scripts/python.exe bench/rife_ram.py <ComfyUI dir> <hires video> <frames> [<frames> ...]

The hires video (768 × 1360) is upscaled ×2 with bicubic instead of Nomos: RIFE's work does not depend on the
content, and this keeps Nomos out of the memory. The node preallocates its float32 output (5 × frames × 3 × 1536 ×
2720 × 4 B, 28.3 GB at 113 frames), so the time per pair shows whether the output still fits in the free RAM.
gc.collect (called by the loop after every pair) and torch.zeros (the output) are timed on their own. Before the
timings, both loops run ACCURACY_FRAMES frames and their outputs are compared.
"""

# pyright: reportMissingImports=false
# ComfyUI's modules (comfy, cv2, the frame-interpolation node...) exist only in ComfyUI's Python, which runs this file

import gc
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import psutil
import torch

COMFY = Path(sys.argv[1])
VIDEO = sys.argv[2]
FRAME_COUNTS = [int(count) for count in sys.argv[3:]]
NODE = COMFY / "custom_nodes" / "comfyui-frame-interpolation"
sys.path[:0] = [str(COMFY), str(NODE)]

import vfi_utils  # noqa: E402
from vfi_models.rife.rife_arch import IFNet  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfyui.rife import interpolate  # noqa: E402

RIFE = NODE / "ckpts" / "rife" / "rife49.pth"
MULTIPLIER = 5
SCALES = [8, 4, 2, 1]
CLEAR_CACHE_AFTER = 50  # the workflow's clear_cache_after_n_frames
ACCURACY_FRAMES = 13
GIB = 2**30


def read_frames(path: str, count: int) -> torch.Tensor:
    """The first `count` frames, ×2 bicubic, as ComfyUI's IMAGE: float32 [count, 1536, 2720, 3] in [0, 1]."""
    capture, frames = cv2.VideoCapture(path), []
    while len(frames) < count:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255
        frames.append(torch.nn.functional.interpolate(rgb, scale_factor=2, mode="bicubic").clamp(0, 1)[0].permute(1, 2, 0))
    return torch.stack(frames)


class MemoryWatch:
    """Samples the free RAM and the pagefile once per second in a thread."""

    def __init__(self):
        self.samples, self.running = [], True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            self.samples.append((psutil.virtual_memory().available, psutil.swap_memory().used))
            time.sleep(1)

    def stop(self) -> str:
        self.running = False
        free = [sample[0] for sample in self.samples]
        paged = [sample[1] for sample in self.samples]
        return (f"free RAM {free[0] / GIB:.1f} -> min {min(free) / GIB:.1f} GiB, pagefile {paged[0] / GIB:.1f} -> max "
                f"{max(paged) / GIB:.1f} GiB")


class Timed:
    """Wraps a function and sums the seconds spent in it."""

    def __init__(self, function):
        self.function, self.seconds, self.calls = function, 0.0, 0

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        result = self.function(*args, **kwargs)
        self.seconds += time.perf_counter() - start
        self.calls += 1
        return result


def node_loop(model, frames: torch.Tensor) -> torch.Tensor:
    """RIFE_VFI.vfi's loop and its output, as ComfyUI gets it."""
    def middle(frame_0, frame_1, timestep, model, scales, fast_mode, ensemble):
        return model(frame_0, frame_1, timestep, scales, fast_mode, ensemble)

    out = vfi_utils.generic_frame_loop("RIFE_VFI", vfi_utils.preprocess_frames(frames), CLEAR_CACHE_AFTER, MULTIPLIER,
                                       middle, model, SCALES, True, False, dtype=torch.float32)
    return vfi_utils.postprocess_frames(out)


def unica_loop(model, frames: torch.Tensor) -> torch.Tensor:
    return interpolate(model, frames, MULTIPLIER, SCALES, True, False, torch.device("cuda"))


LOOPS = {"node": node_loop, "unica": unica_loop}


def run(name: str, model, frames: torch.Tensor, report: bool = True) -> torch.Tensor | None:
    collect, zeros = Timed(gc.collect), Timed(torch.zeros)
    vfi_utils.gc.collect, vfi_utils.torch.zeros = collect, zeros  # vfi_utils uses the modules gc and torch
    watch = MemoryWatch()
    process = psutil.Process()
    faults = process.memory_info().num_page_faults
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = LOOPS[name](model, frames)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    vfi_utils.gc.collect, vfi_utils.torch.zeros = gc.collect, torch.zeros
    memory = watch.stop()
    if not report:
        return out
    pairs = frames.shape[0] - 1
    print(f"{name:5} {frames.shape[0]:3} frames, output {out.numel() * 4 / GIB:.1f} GiB: {seconds:.1f} s, "
          f"{seconds / pairs:.3f} s per pair ({seconds / pairs * 112:.0f} s for 112); torch.zeros {zeros.seconds:.1f} s, "
          f"gc.collect {collect.seconds:.1f} s in {collect.calls} calls; page faults "
          f"+{(process.memory_info().num_page_faults - faults) / 1e6:.1f} M; {memory}", flush=True)
    return None


def compare(model, frames: torch.Tensor) -> None:
    node, unica = (run(name, model, frames, report=False) for name in LOOPS)
    mse = (unica - node).pow(2).mean().item()
    print(f"{frames.shape[0]} frames: shapes {tuple(node.shape)} / {tuple(unica.shape)}, largest difference "
          f"{(unica - node).abs().max().item():.2e}, PSNR {10 * math.log10(1 / mse) if mse else math.inf:.1f} dB, "
          f"unica contiguous {unica.is_contiguous()}, node contiguous {node.is_contiguous()}", flush=True)


def main() -> None:
    torch.backends.cudnn.benchmark = False  # ComfyUI's default
    print(f"free RAM at start {psutil.virtual_memory().available / GIB:.1f} GiB, threads {torch.get_num_threads()}")
    with torch.inference_mode():
        rife = IFNet(arch_ver="4.7")
        rife.load_state_dict(torch.load(RIFE))
        rife.eval().cuda()
        compare(rife, read_frames(VIDEO, ACCURACY_FRAMES))
        for count in FRAME_COUNTS:
            frames = read_frames(VIDEO, count)
            for name in LOOPS:
                run(name, rife, frames[:2], report=False)  # warm-up
                run(name, rife, frames)
                gc.collect()
            del frames
            gc.collect()


if __name__ == "__main__":
    main()
