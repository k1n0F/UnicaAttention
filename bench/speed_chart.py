"""
Speed of every attention kernel over sequence length, as a bar chart for the README.

    uv run --with matplotlib python bench/speed_chart.py [--heads 40] [--runs 5]

The kernels and the timing are bench/harness.py's (whole call, quantization included; one warm-up, then the
median). Throughput is 4 · heads · N² · 128 operations per call, non-causal. Writes build/speed_chart.json and
assets/speed.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from harness import kernels, time_ms

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = [1024, 2048, 4096, 8192, 16384, 32760]  # 32 760: WAN 2.2, 81 frames at 832×480
HEAD_DIM = 128
COLORS = {"PyTorch SDPA": "#5b6577", "SageAttention 2": "#8ea3c0", "SageAttention 3": "#c9a46a",
          "UnicaAttention": "#66e5c2"}


def measure(heads: int, runs: int) -> dict[str, dict[int, float]]:
    """TOPS per kernel and sequence length."""
    tops: dict[str, dict[int, float]] = {}
    for n in LENGTHS:
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, heads, n, HEAD_DIM, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        for name, kernel in kernels().items():
            ms = time_ms(lambda kernel=kernel: kernel(q, k.clone(), v), runs)  # SageAttention 3 modifies k
            tops.setdefault(name, {})[n] = 4 * heads * n * n * HEAD_DIM / ms / 1e9
            print(f"{n:6} {name:16} {ms:9.2f} ms {tops[name][n]:6.0f} TOPS", flush=True)
    return tops


def plot(tops: dict[str, dict[int, float]], heads: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(tops)
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    for i, name in enumerate(names):
        xs = [j + (i - (len(names) - 1) / 2) * width for j in range(len(LENGTHS))]
        values = [tops[name][n] for n in LENGTHS]
        bars = ax.bar(xs, values, width, label=name, color=COLORS.get(name, "#999999"),
                      edgecolor="#0d1117", linewidth=0.6)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6, f"{value:.0f}", ha="center",
                    va="bottom", fontsize=7, color="#c9d1d9", rotation=90)
    ax.set_xticks(range(len(LENGTHS)), [f"{n // 1024}K" if n % 1024 == 0 else f"{n:,}".replace(",", " ")
                                         for n in LENGTHS])
    ax.set_xlabel("Sequence length", color="#c9d1d9")
    ax.set_ylabel("Speed (TOPS)", color="#c9d1d9")
    ax.set_title(f"RTX 5080, head dim 128, {heads} heads, non-causal, whole call", color="#e6edf3", fontsize=11)
    ax.set_ylim(0, max(max(t.values()) for t in tops.values()) * 1.22)
    ax.tick_params(colors="#c9d1d9")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#30363d")
    ax.grid(axis="y", color="#21262d", linewidth=0.6)
    ax.set_axisbelow(True)
    legend = ax.legend(ncol=len(names), loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("#c9d1d9")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=40)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    results = measure(args.heads, args.runs)
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "speed_chart.json").write_text(json.dumps({"heads": args.heads, "tops": results}, indent=2))
    plot(results, args.heads, ROOT / "assets" / "speed.png")
