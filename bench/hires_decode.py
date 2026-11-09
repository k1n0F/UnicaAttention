"""M4.40: the hires decode of "wan stable work" (node 981) as the node runs it. WanVideoWrapper's own loader, encode
and decode nodes are called. Run it with ComfyUI's Python:

    <ComfyUI>/.venv-13/Scripts/python.exe bench/hires_decode.py <ComfyUI dir> whole|tiled [ballast GiB ...]
    <ComfyUI>/.venv-13/Scripts/python.exe bench/hires_decode.py <ComfyUI dir> quality <hires video> [tiling names]

whole / tiled: a latent of the hires pass's shape with random values (a convolution's time does not depend on them),
decoded while a ballast tensor holds part of the VRAM. For each run: the free VRAM before the decode (the ballast in
place), the decode's own peak (torch reserved above what it was before), the part of that peak the free VRAM could not
hold (what Windows' sysmem fallback places in shared memory), and the time. The first run warms cuDNN up.

quality: a real hires video encoded into a latent, decoded whole (the reference) and with each tile setting in
TILINGS. PSNR over the whole video, and in the worst 16-pixel band of columns and of rows: a seam is an error gathered
along a tile's edge, which the whole-video PSNR averages away. Frame FRAME of each decode is saved to build/.
"""

# pyright: reportMissingImports=false
# ComfyUI's modules (comfy, folder_paths, WanVideoWrapper, cv2) exist only in ComfyUI's Python, which runs this file

import importlib
import sys
import time
from pathlib import Path

import torch

COMFY = Path(sys.argv[1])
MODE = sys.argv[2]
sys.path[:0] = [str(COMFY), str(COMFY / "custom_nodes")]

import cv2  # noqa: E402
import server  # noqa: E402

server.PromptServer.instance = None  # the package reads it on import (for previews); no server runs here
nodes = importlib.import_module("ComfyUI-WanVideoWrapper.nodes")  # noqa: E402
loading = importlib.import_module("ComfyUI-WanVideoWrapper.nodes_model_loading")  # noqa: E402

VAE = "Wan2.1_VAE.pth"  # the workflow's WanVideoVAELoader 957: bf16, no CPU cache
LATENT = (1, 16, 29, 170, 96)  # 113 portrait frames, 1360 high and 768 wide: the hires pass's 118,320 tokens
TILES = dict(tile_x=272, tile_y=272, tile_stride_x=144, tile_stride_y=128)  # the node's defaults
TILINGS = {  # tile and stride in pixels, x then y
    "272 / 144 (default)": TILES,
    "512 / 256": dict(tile_x=512, tile_y=512, tile_stride_x=256, tile_stride_y=256),
    # tiles as wide as the frame (768), stacked: two with a 16 px overlap, two with 176 px, three with 88 px
    "two tiles": dict(tile_x=896, tile_y=768, tile_stride_x=464, tile_stride_y=752),
    "two tiles, wide": dict(tile_x=896, tile_y=768, tile_stride_x=464, tile_stride_y=592),
    "three tiles": dict(tile_x=896, tile_y=512, tile_stride_x=464, tile_stride_y=424),
}
ONLY = sys.argv[4:]  # quality: run only the tilings whose names start with these words
BAND = 16  # pixels
FRAME = 56  # the middle of 113
OUT = Path(__file__).resolve().parents[1] / "build"
GIB = 2**30


def decode(vae, latent: torch.Tensor, tiled: bool, tiles: dict, ballast_gib: float = 0):
    """The node's decode with a ballast in the VRAM: the images (None on OOM) and a line of numbers."""
    ballast = torch.ones(int(ballast_gib * GIB), dtype=torch.uint8, device="cuda")  # written, so it is resident
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    reserved_before = torch.cuda.memory_reserved()
    free_before = torch.cuda.mem_get_info()[0]
    start = time.perf_counter()
    try:
        images = nodes.WanVideoDecode().decode(vae, {"samples": latent}, tiled, **tiles)[0]
        seconds = f"{time.perf_counter() - start:6.1f} s"
    except torch.OutOfMemoryError:
        images, seconds = None, "   OOM"
    peak = torch.cuda.max_memory_reserved() - reserved_before
    del ballast
    torch.cuda.empty_cache()
    return images, (f"ballast {ballast_gib:4.1f} GiB  free {free_before / GIB:5.2f}  peak {peak / GIB:5.2f}  "
                    f"over {max(0, peak - free_before) / GIB:5.2f} GiB  {seconds}")


def psnr(mse: torch.Tensor) -> float:
    return 10 * torch.log10(1 / mse.clamp_min(1e-12)).item()


def worst_band(error: torch.Tensor, dim: int) -> float:
    """PSNR of the worst BAND-wide band of columns (dim 2) or rows (dim 1) of a [frames, h, w, 3] squared error."""
    profile = error.mean(dim=[d for d in range(4) if d != dim])
    return psnr(profile[: len(profile) // BAND * BAND].view(-1, BAND).mean(1).max())


def read_video(path: str) -> torch.Tensor:
    capture, frames = cv2.VideoCapture(path), []
    while (frame := capture.read()[1]) is not None:
        frames.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    return torch.stack(frames).float() / 255


def save(images: torch.Tensor, name: str):
    OUT.mkdir(exist_ok=True)
    frame = (images[FRAME].clamp(0, 1) * 255).round().byte().numpy()
    cv2.imwrite(str(OUT / f"m440_{name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def quality(vae, video: str):
    frames = read_video(video)
    print(f"{video}: {tuple(frames.shape)}", flush=True)
    latent = nodes.WanVideoEncode().encode(vae, frames, True, **TILES)[0]["samples"]  # tiled: only a real latent
    reference, line = decode(vae, latent, False, TILES)
    print(f"{'whole (reference)':20} {line}", flush=True)
    save(reference, "whole")
    for name, tiles in TILINGS.items():
        if ONLY and not any(name.startswith(word) for word in ONLY):
            continue
        images, line = decode(vae, latent, True, tiles)
        error = (images - reference).square()
        print(f"{name:20} {line}  PSNR {psnr(error.mean()):5.1f} dB, worst column band "
              f"{worst_band(error, 2):5.1f}, worst row band {worst_band(error, 1):5.1f}", flush=True)
        short = name.split(" (")[0].replace(" / ", "_").replace(" ", "_")
        save(images, short)
        save((images - reference).abs() * 8, "diff8_" + short)  # the error, 8x brighter


def main():
    print(f"torch {torch.__version__}, cudnn.benchmark {torch.backends.cudnn.benchmark}", flush=True)
    vae = loading.WanVideoVAELoader().loadmodel(VAE, "bf16")[0]
    if MODE == "quality":
        return quality(vae, sys.argv[3])
    tiled = MODE == "tiled"
    latent = torch.randn(LATENT, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    print("warm-up:", decode(vae, latent, tiled, TILES)[1], flush=True)
    for ballast_gib in [float(gib) for gib in sys.argv[3:]] or [0, 4, 6, 8, 10, 12]:
        print(MODE, decode(vae, latent, tiled, TILES, ballast_gib)[1], flush=True)


if __name__ == "__main__":
    main()
