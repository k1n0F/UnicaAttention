"""M4.33: microbenchmarks on the call signatures of the WAN workflows (self-attention, 40 heads, d = 128, fp16 in
ComfyUI's [batch, tokens, heads, 128]), one part per hypothesis of LOG.md M4.33.

usage: uv run python bench/signatures.py mean          H1: K's mean, rope_stats against torch's k.mean
       uv run python bench/signatures.py speed         H2: whole call per mode and length, TOPS
       uv run python bench/signatures.py pieces [n]    H3: 118,320 (or n) tokens in pieces of 1 head against 10
       uv run python bench/signatures.py ripple [set]  H4: where fast's error sits along the tokens (dumps: bench,
                                                        stable, hires)
"""

import statistics
import sys
from pathlib import Path

import torch
from emulate import errors, reference
from sageattention import sageattn
from sageattn3 import sageattn3_blackwell

import unica
from unica.__main__ import time_ms

REPO = Path(__file__).resolve().parents[1]
LENGTHS = {"stable work, high / Lightning": 29928, "Rigsy bench": 34020, "stable work, hires": 118320}
HEADS, DIM = 40, 128
MODES = ("per_channel", "per_token", "fast")
PEAK = {"per_channel": 460e12, "per_token": 460e12, "fast": 935e12, "sage2": 460e12, "sage3": 935e12}
# Dumps by signature: file pattern, tokens, tokens per row and per frame (latent / 8, patches of 2 × 2)
DUMPS = {"bench": ("wan_call*.pt", 34020, 54, 1620),  # Rigsy bench, 480 × 864: 30 × 54 per frame (M4.22)
         "stable": ("WanVideo_call*.pt", 29928, 43, 1032),  # stable work high / Lightning, 384 × 688: 24 × 43
         "hires": ("WanVideo_call*.pt", 118320, 85, 4080)}  # stable work hires, 768 × 1360: 48 × 85 (M4.34)
SMOOTH = 512


def nhd(tokens: int, heads: int = HEADS) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(1, tokens, heads, DIM, device="cuda", dtype=torch.float16) for _ in range(3)]


def unica_nhd(mode: str):
    """The node's call: [batch, tokens, heads, 128] seen as [batch, heads, tokens, 128], fp16 in and out."""
    return lambda q, k, v: unica.attention(*(x.transpose(1, 2) for x in (q, k, v)), **unica.PRESETS[mode])


def torch_mean(original):
    """rope_stats with torch's k.mean as its mean: the accurate mode's K mean before M4.33."""
    def stats(x, rope, per=None):
        low_high = original(x, rope, per)[1:]
        return (x.mean(-2, dtype=torch.float32) if per is None and rope is None else original(x, rope, per)[0]), *low_high
    return stats


def mean() -> None:
    original, original_heads = unica.rope_stats, unica.heads_per_call
    for name, tokens in LENGTHS.items():
        k = torch.randn(1, 12, tokens, DIM, device="cuda", dtype=torch.float16).bfloat16()
        ms_torch = time_ms(lambda k=k: k.mean(-2, dtype=torch.float32), runs=20)
        ms_own = time_ms(lambda k=k: original(k, None)[0], runs=20)
        print(f"{name}, {tokens} tokens, 12 heads: k.mean {ms_torch:.3f} ms, rope_stats {ms_own:.3f} ms")
        q, v = (torch.randn_like(k) for _ in range(2))
        for label, stats in (("k.mean", torch_mean(original)), ("rope_stats", original)):
            unica.rope_stats = stats
            whole = None
            for step in (12, 4, 1):
                unica.heads_per_call = lambda *args, step=step: step
                out = unica.attention(q, k, v, **unica.PRESETS["per_token"])
                whole = out if whole is None else whole
                if step != 12:
                    print(f"  {label:10} pieces of {step:2} == one call of 12: {torch.equal(out, whole)}")
        unica.rope_stats, unica.heads_per_call = original, original_heads
    # the error on the real WAN dumps, both K means
    rows = []
    for path in sorted((REPO / "build" / "dumps").glob("wan_call*.pt")):
        d = torch.load(path, map_location="cuda", weights_only=False)
        q, k, v = (d[x].bfloat16() for x in "qkv")
        ref = reference(q, k, v)
        before = errors(_with(torch_mean(original), q, k, v).float(), ref)["rel_l2"]
        after = errors(unica.attention(q, k, v, **unica.PRESETS["per_token"]).float(), ref)["rel_l2"]
        rows.append((before, after))
    diffs = [abs(a - b) * 100 for b, a in rows]
    print(f"WAN dumps ({len(rows)}): per token rel L2 k.mean {100 * statistics.mean(b for b, _ in rows):.4f} %, "
          f"rope_stats {100 * statistics.mean(a for _, a in rows):.4f} %, |change| max {max(diffs):.5f} pp")


def _with(stats, q, k, v):
    original = unica.rope_stats
    unica.rope_stats = stats
    try:
        return unica.attention(q, k, v, **unica.PRESETS["per_token"])
    finally:
        unica.rope_stats = original


def speed() -> None:
    paths = {**{m: unica_nhd(m) for m in MODES},
             "sage2": lambda q, k, v: sageattn(q, k, v, tensor_layout="NHD"),
             "sage3": lambda q, k, v: sageattn3_blackwell(*(x.transpose(1, 2) for x in (q, k, v))).transpose(1, 2)}
    for name, tokens in LENGTHS.items():
        q, k, v = nhd(tokens)
        work = 4 * tokens**2 * DIM * HEADS
        times = {p: [] for p in paths if not (p == "sage3" and tokens > 60000)}  # SA3 needs 27.9 GiB at 118 k (M4.31)
        runs = 3 if tokens > 60000 else 5
        for r in range(3):
            for p in (times if r % 2 == 0 else reversed(list(times))):
                times[p].append(time_ms(lambda p=p, qkv=(q, k, v): paths[p](*qkv), runs=runs))
        print(f"{name}, {tokens} tokens ({work / 1e12:.1f} TOP per call):")
        for p, t in times.items():
            ms = statistics.median(t)
            tops = work / ms * 1e3
            print(f"  {p:12} {ms:8.2f} ms  {tops / 1e12:6.1f} TOPS  {100 * tops / PEAK[p]:5.1f} % of peak")
        del q, k, v  # before the next length's inputs
        torch.cuda.empty_cache()


def pieces(tokens: int = LENGTHS["stable work, hires"]) -> None:
    """H3, and H3b: a token count given on the command line, e.g. 118272 (1848 blocks of 64, exactly 11 waves)."""
    q, k, v = nhd(tokens)
    original = unica.heads_per_call
    for mode in ("per_token", "fast"):
        call = unica_nhd(mode)
        times = {1: [], 10: []}
        for r in range(3):
            for step in ((1, 10) if r % 2 == 0 else (10, 1)):
                unica.heads_per_call = lambda *args, step=step: step
                times[step].append(time_ms(lambda call=call: call(q, k, v), runs=3))
        one, ten = statistics.median(times[1]), statistics.median(times[10])
        print(f"{mode:10} {tokens} tokens: pieces of 10 heads {ten:7.1f} ms, of 1 head {one:7.1f} ms ({100 * (one / ten - 1):+.1f} %)")
    unica.heads_per_call = original


def ripple(dumps: str = "bench") -> None:
    pattern, tokens, row, frame = DUMPS[dumps]
    print(f"{dumps}: {tokens} tokens, {row} per row, {frame} per frame")
    edges, middles, peaks = {m: [] for m in ("per_token", "fast")}, {m: [] for m in ("per_token", "fast")}, []
    profiles = {}
    for path in sorted((REPO / "build" / "dumps").glob(pattern)):
        d = torch.load(path, map_location="cuda", weights_only=False)
        if d["q"].shape[2] != tokens:
            continue
        q, k, v = (d[x].bfloat16() for x in "qkv")
        ref = reference(q, k, v)
        per_query = {}
        for mode in ("per_token", "fast"):
            out = unica.attention(q, k, v, **unica.PRESETS[mode]).float()
            e = ((out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)).mean((0, 1))  # [tokens]
            per_query[mode] = e
            n = e.shape[0] // SMOOTH * SMOOTH
            blocks = e[:n].view(-1, SMOOTH)
            edges[mode].append(torch.cat([blocks[:, :16], blocks[:, -16:]], 1).mean().item())
            middles[mode].append(blocks[:, 16:-16].mean().item())
        ratio = per_query["fast"] / per_query["per_token"].clamp_min(1e-12)
        for period in (SMOOTH, 504):  # 504: a control period with no meaning in the kernel or the frame
            n = ratio.shape[0] // period * period
            profiles.setdefault(period, []).append(ratio[:n].view(-1, period).mean(0))
        spectrum = torch.fft.rfft(ratio - ratio.mean()).abs()
        frequencies = torch.fft.rfftfreq(ratio.shape[0], device=ratio.device)
        median = spectrum[1:].median()
        for period in (row, SMOOTH, frame):
            i = int(torch.argmin((frequencies - 1 / period).abs()))
            peaks.append((period, (spectrum[max(1, i - 2):i + 3].max() / median).item()))
        top = int(torch.argmax(spectrum[1:])) + 1
        peaks.append(("top", 1 / frequencies[top].item(), (spectrum[top] / median).item()))
    for mode, values in edges.items():
        e, m = statistics.mean(values), statistics.mean(middles[mode])
        print(f"{mode:10} per-query error: block edges {100 * e:.3f} %, middles {100 * m:.3f} % (edges/middle {e / m:.3f})")
    for period in (row, SMOOTH, frame):
        values = [p[1] for p in peaks if p[0] == period]
        print(f"fast/accurate spectrum at period {period:5}: peak/median median {statistics.median(values):.2f}, "
              f"max {max(values):.2f}")
    for period, rows in profiles.items():  # fast / accurate by position within the period, in eighths
        profile = torch.stack(rows).mean(0)
        eighths = profile.view(8, -1).mean(1) / profile.mean()
        print(f"fast/accurate by position within {period:3} queries (eighths, / mean): "
              + " ".join(f"{x:.3f}" for x in eighths.tolist()))
    tops = [p for p in peaks if p[0] == "top"]
    print("strongest periods per dump (tokens, peak/median):",
          ", ".join(f"{t[1]:.0f}/{t[2]:.1f}" for t in tops))


if __name__ == "__main__":
    {"mean": mean, "speed": speed, "pieces": pieces, "ripple": ripple}[sys.argv[1]](*(int(a) if a.isdigit() else a for a in sys.argv[2:]))
