"""M4.35: the tail of "wan stable work", Nomos ×2 then RIFE ×5, as ComfyUI runs them and as they could run. Uses
ComfyUI's own code (comfy.utils.tiled_scale, the frame-interpolation node's loop and IFNet), so run it with ComfyUI's
Python:

    <ComfyUI>/.venv/Scripts/python.exe bench/rife_loop.py <ComfyUI dir> <hires video> [frames, default 13]

The video is the hires pass before Nomos (768 × 1360). Its first frames go through Nomos, and Nomos's frames through
RIFE; times per frame and per pair scale to the render's 113 frames and 112 pairs.
"""

import gc
import math
import sys
import time
from pathlib import Path

import cv2
import torch

COMFY = Path(sys.argv[1])
VIDEO = sys.argv[2]
FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 13
NODE = COMFY / "custom_nodes" / "comfyui-frame-interpolation"
sys.path[:0] = [str(COMFY), str(NODE)]

import comfy.utils  # noqa: E402
from spandrel import ModelLoader  # noqa: E402
from vfi_models.rife.rife_arch import IFNet  # noqa: E402
from vfi_utils import generic_frame_loop, postprocess_frames, preprocess_frames  # noqa: E402

NOMOS = COMFY / "models" / "upscale_models" / "2xNomosUni_span_multijpg_ldl.safetensors"
RIFE = NODE / "ckpts" / "rife" / "rife49.pth"
MULTIPLIER = 5  # the workflow's RIFE settings: rife49, ×5, fast_mode on, ensemble off, scale 1
SCALES = [8, 4, 2, 1]
RENDER_FRAMES = 113


def read_frames(path: str, n: int) -> torch.Tensor:
    """The first n frames, as ComfyUI's IMAGE: float32 [n, height, width, 3] in [0, 1]."""
    capture, frames = cv2.VideoCapture(path), []
    while len(frames) < n:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return torch.stack(frames).float() / 255


def timed(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean().item()
    return math.inf if mse == 0 else 10 * math.log10(1 / mse)


def nomos_node(model, image: torch.Tensor) -> torch.Tensor:
    """comfy_extras/nodes_upscale_model.py ImageUpscaleWithModel: all frames to the GPU, fp32 tiles of 512 with an
    overlap of 32, the result on the CPU in float32."""
    s = comfy.utils.tiled_scale(image.movedim(-1, -3).cuda(), lambda a: model(a.float()), tile_x=512, tile_y=512,
                                overlap=32, upscale_amount=model.scale, output_device="cpu")
    return torch.clamp(s.movedim(-3, -1), min=0, max=1.0)


def nomos_whole(model, image: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """One frame at a time, no tiles, in dtype."""
    out = torch.empty(image.shape[0], image.shape[1] * 2, image.shape[2] * 2, 3, pin_memory=True)
    for i in range(image.shape[0]):
        x = image[i:i + 1].movedim(-1, -3).cuda().to(dtype)
        out[i].copy_(model(x).clamp(0, 1)[0].movedim(0, -1).float(), non_blocking=True)
    return out


def rife_node(model, frames: torch.Tensor) -> torch.Tensor:
    """The node's call (RIFE_VFI.vfi): generic_frame_loop with the node's argument order."""
    def middle(f0, f1, t, model, scales, fast_mode, ensemble):
        return model(f0, f1, t, scales, fast_mode, ensemble)
    out = generic_frame_loop("RIFE_VFI", preprocess_frames(frames), 50, MULTIPLIER, middle, model, SCALES, True, False,
                             dtype=torch.float32)
    return postprocess_frames(out)


def rife_batched(model, frames: torch.Tensor, autocast: bool) -> torch.Tensor:
    """Each frame uploaded once, a pair's 4 middle frames in one batch, copied back to pinned memory without a sync,
    no gc.collect. The model is called with the node's arguments (fast_mode lands in `training`), so in fp32 the
    frames should be the node's."""
    n, steps = frames.shape[0], MULTIPLIER - 1
    chw = preprocess_frames(frames)  # [n, 3, h, w], a view
    out = torch.empty((n - 1) * MULTIPLIER + 1, *chw.shape[1:], pin_memory=True)
    t = (torch.arange(1, MULTIPLIER, device="cuda", dtype=torch.float32) / MULTIPLIER).view(-1, 1, 1, 1)
    f1 = chw[0:1].cuda()
    for i in range(n - 1):
        f0, f1 = f1, chw[i + 1:i + 2].cuda()
        base = i * MULTIPLIER
        out[base].copy_(chw[i])
        with torch.autocast("cuda", torch.float16, enabled=autocast):
            middle = model(f0.expand(steps, -1, -1, -1), f1.expand(steps, -1, -1, -1), t, SCALES, True, False)
        out[base + 1:base + MULTIPLIER].copy_(middle.float(), non_blocking=True)
    out[-1].copy_(chw[-1])
    torch.cuda.synchronize()
    return postprocess_frames(out)


def encode(frames: torch.Tensor, ffmpeg: str) -> tuple[float, float]:
    """VHS VideoCombine, video/h264-mp4 at 60 fps: each float32 frame to bytes with numpy (its tensor_to_bytes), piped
    to ffmpeg with its arguments. Returns (seconds converting alone, seconds converting and encoding as VHS does)."""
    import subprocess
    import tempfile

    import numpy as np

    def to_bytes(frame):  # videohelpersuite/nodes.py tensor_to_int(tensor, 8).astype(np.uint8)
        return np.clip(frame.cpu().numpy() * 255 + 0.5, 0, 255).astype(np.uint8).tobytes()

    start = time.perf_counter()
    for frame in frames:
        to_bytes(frame)
    converting = time.perf_counter() - start
    height, width = frames.shape[1:3]
    with tempfile.TemporaryDirectory() as folder:
        args = [ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-color_range", "pc", "-colorspace", "rgb",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-s", f"{width}x{height}", "-r", "60", "-i", "-",
                "-n", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-vf", "scale=out_color_matrix=bt709",
                "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                str(Path(folder) / "out.mp4")]
        start = time.perf_counter()
        with subprocess.Popen(args, stdin=subprocess.PIPE) as process:
            for frame in frames:
                process.stdin.write(to_bytes(frame))
            process.stdin.close()
            process.wait()
        both = time.perf_counter() - start
    return converting, both


def main() -> None:
    torch.backends.cudnn.benchmark = False  # ComfyUI's default
    image = read_frames(VIDEO, FRAMES)
    n = image.shape[0]
    print(f"{n} frames of {image.shape[2]} × {image.shape[1]} from the hires video")
    with torch.inference_mode():
        nomos = ModelLoader().load_from_state_dict(comfy.utils.load_torch_file(str(NOMOS))).eval().to("cuda")
        nomos_node(nomos, image[:1])  # warm-up
        up, seconds = timed(lambda: nomos_node(nomos, image))
        print(f"Nomos, as the node:        {seconds / n:6.3f} s per frame, {seconds / n * RENDER_FRAMES:6.1f} s for "
              f"{RENDER_FRAMES} frames")
        for dtype in (torch.float32, torch.float16):
            nomos.to(dtype)
            nomos_whole(nomos, image[:1], dtype)
            other, seconds = timed(lambda dtype=dtype: nomos_whole(nomos, image, dtype))
            print(f"Nomos, whole frames, {str(dtype)[6:]}: {seconds / n:6.3f} s per frame, PSNR against the node "
                  f"{psnr(other, up):.1f} dB")
        nomos.to(torch.float32)
        del nomos
        torch.cuda.empty_cache()

        rife = IFNet(arch_ver="4.7")
        rife.load_state_dict(torch.load(RIFE))
        rife.eval().cuda()
        pairs = n - 1
        rife_node(rife, up[:2])  # warm-up
        node, seconds = timed(lambda: rife_node(rife, up))
        node_pair = seconds / pairs
        print(f"RIFE ×5, as the node:      {node_pair:6.3f} s per pair, {node_pair * (RENDER_FRAMES - 1):6.1f} s for "
              f"{RENDER_FRAMES - 1} pairs ({node.shape[0]} frames out)")
        for autocast in (False, True):
            rife_batched(rife, up[:2], autocast)
            mine, seconds = timed(lambda autocast=autocast: rife_batched(rife, up, autocast))
            label = "fp16 autocast" if autocast else "fp32"
            print(f"RIFE ×5, batched, {label:13}: {seconds / pairs:6.3f} s per pair ({node_pair * pairs / seconds:.2f}× "
                  f"the node), largest difference {(mine - node).abs().max().item():.2e}, PSNR {psnr(mine, node):.1f} dB")
        import imageio_ffmpeg  # VHS's default ffmpeg
        out_frames = (RENDER_FRAMES - 1) * MULTIPLIER + 1
        converting, both = encode(node, imageio_ffmpeg.get_ffmpeg_exe())
        print(f"VideoCombine, {node.shape[0]} frames: to bytes {converting / node.shape[0]:.3f} s per frame, with x264 "
              f"{both / node.shape[0]:.3f} s per frame ({node.shape[0] / both:.1f} fps), {both / node.shape[0] * out_frames:.1f} s "
              f"for {out_frames} frames")
        start = time.perf_counter()
        gc.collect()
        print(f"one gc.collect() in this process: {1e3 * (time.perf_counter() - start):.1f} ms "
              f"(the node calls it once per pair, in ComfyUI's much larger heap)")


if __name__ == "__main__":
    main()
