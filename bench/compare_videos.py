"""The same render by several attention modes, frame by frame against a reference: what the attention did to the video.

usage: uv run python bench/compare_videos.py reference.mp4 other.mp4 [...] [--sheet sheet.png] [--crop x,y,w,h]

Per video, against the reference:
- PSNR per frame (RGB, 8 bit): the mean and the worst frame;
- flicker: the mean |(f[t] - f[t-1]) - (ref[t] - ref[t-1])|, how much the change between neighbouring frames differs
  from the reference's, in 8-bit levels. A mode that adds trembling raises it even where its single frames look alike;
  its own motion, |f[t] - f[t-1]|, is printed next to the reference's.
--sheet: a picture with a row per video and a column per moment (0, 1/4, 1/2, 3/4, end), to look at; --crop takes the
same box out of every frame first (a face, a hand), in the video's pixels.
Needs ffmpeg on PATH; runs on the CPU.
"""

import argparse
import json
import subprocess
from pathlib import Path

import torch

MOMENTS = (0.0, 0.25, 0.5, 0.75, 1.0)
SHEET_WIDTH = 360  # pixels per cell, before the crop is scaled to it


def frames(path: Path) -> torch.Tensor:
    """All frames as uint8 [n, height, width, 3]."""
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=width,height", "-of", "json", str(path)],
                                      capture_output=True, check=True).stdout)["streams"][0]
    width, height = probe["width"], probe["height"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(-1, height, width, 3)


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per frame, in dB."""
    mse = (a.float() - b.float()).pow(2).flatten(1).mean(1)
    return 10 * torch.log10(255**2 / mse.clamp_min(1e-10))


def flicker(video: torch.Tensor, reference: torch.Tensor) -> float:
    step, step_ref = video[1:].float() - video[:-1].float(), reference[1:].float() - reference[:-1].float()
    return (step - step_ref).abs().mean().item()


def motion(video: torch.Tensor) -> float:
    return (video[1:].float() - video[:-1].float()).abs().mean().item()


def sheet(videos: dict[str, torch.Tensor], out: Path, crop: tuple[int, int, int, int] | None) -> None:
    import torch.nn.functional as F
    rows = []
    for video in videos.values():
        cells = []
        for moment in MOMENTS:
            frame = video[round(moment * (len(video) - 1))]
            if crop:
                x, y, w, h = crop
                frame = frame[y:y + h, x:x + w]
            frame = frame.permute(2, 0, 1)[None].float()
            height = round(frame.shape[2] * SHEET_WIDTH / frame.shape[3])
            cells.append(F.interpolate(frame, size=(height, SHEET_WIDTH), mode="area")[0])
        rows.append(torch.cat(cells, 2))
    picture = torch.cat(rows, 1).clamp(0, 255).byte().permute(1, 2, 0).contiguous()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                    f"{picture.shape[1]}x{picture.shape[0]}", "-i", "-", str(out)],
                   input=bytes(picture.untyped_storage()), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("others", type=Path, nargs="+")
    parser.add_argument("--sheet", type=Path)
    parser.add_argument("--crop", help="x,y,w,h")
    args = parser.parse_args()
    videos = {path.stem: frames(path) for path in [args.reference, *args.others]}
    reference = videos[args.reference.stem]
    print(f"{args.reference.stem}: {tuple(reference.shape)}, own motion {motion(reference):.2f} levels per frame")
    for name, video in list(videos.items())[1:]:
        if video.shape != reference.shape:
            print(f"{name}: shape {tuple(video.shape)} differs from the reference's, skipped")
            continue
        p = psnr(video, reference)
        print(f"{name}: PSNR mean {p.mean():.2f} dB, worst {p.min():.2f} dB (frame {p.argmin().item()}); "
              f"flicker {flicker(video, reference):.2f} levels; own motion {motion(video):.2f}")
    if args.sheet:
        sheet(videos, args.sheet, tuple(map(int, args.crop.split(","))) if args.crop else None)
        print(f"sheet: {args.sheet}")


if __name__ == "__main__":
    main()
