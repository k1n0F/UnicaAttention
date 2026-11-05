"""Five real models in ComfyUI, a whole render each: ComfyUI's own attention and SageAttention 2 and 3 against
Unica's accurate mode (three V scales) and its fast mode.

usage: uv run python bench/comfyui_models.py [model ...] [--rounds N]     models: see MEASURE_EVERY (default: all)

ComfyUI must be running at URL with the Unica node installed from this repository (custom_nodes/UnicaAttention linked
to comfyui/), KJNodes and SageAttention 2 and 3 installed, and the models of bench/comfyui/models.md in place. The
graphs are bench/comfyui/<model>.json, ComfyUI's API format with fixed seeds, prompts, sizes and steps; this script
sets only the Unica nodes' inputs, and puts a SageAttention patch before them for the sage modes.

Per model: free ComfyUI's memory (otherwise WDDM pages, LOG M4.16), one measured run (every n-th self-attention call
compared with the exact attention, for every mode at once; it also loads the models), then ROUNDS rounds of the
MODES on the warm models, in alternating order. Every row of OUT_CSV carries the environment it ran in; at the end,
the median time per mode and each mode's error on the measured calls."""
import argparse
import csv
import json
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

URL = "http://127.0.0.1:8188"
REPO = Path(__file__).resolve().parents[1]
GRAPHS = REPO / "bench" / "comfyui"
INPUT_IMAGE = GRAPHS / "unica_bench.png"  # LoadImage's "unica_bench.png" in the graphs of wan and qwen_edit
ERRORS_CSV = REPO / "build" / "attention_errors_v_modes.csv"  # what the node writes on its measured calls
OUT_CSV = REPO / "build" / "comfyui_models.csv"
# the measured run compares every n-th self-attention call: about 20-30 calls per model
MEASURE_EVERY = {"wan": 7, "ltx23": 26, "qwen_edit": 12, "zimage": 42, "moss": 63}
# off: the Unica node passes the model on untouched, so ComfyUI's own attention runs (SDPA without launch flags).
# sage2, sage3: the same, with KJNodes' PathchSageAttentionKJ before it, which sends every attention call to
# SageAttention ("auto" picks SageAttention 2's kernel for the GPU). fast: Unica's fast mode (M4.21).
MODES = ("off", "sage2", "sage3", "per_channel", "per_token", "outliers", "fast")
SAGE = {"sage2": "auto", "sage3": "sageattn3"}  # PathchSageAttentionKJ's sage_attention
# the node's measurement columns (comfyui/__init__.py) per mode, and the rival each mode is held against
ERROR_COLUMNS = {"sage2": "sage2", "sage3": "sage3", "per_channel": "accurate_per_channel",
                 "per_token": "accurate_per_token", "outliers": "accurate_outliers", "fast": "fp4_smooth512"}
RIVAL = {"per_channel": "sage2", "per_token": "sage2", "outliers": "sage2", "fast": "sage3"}
ROUNDS = 3  # a single LTX render varies by +-5 s
POLL_SECONDS = 2
RESTART_WAIT_SECONDS = 600  # ComfyUI's watchdog restarts it after a crash
FIELDS = ["time", "model", "mode", "round", "seconds", "calls", "rel_l2_median", "rel_l2_max", "rival",
          "rival_rel_l2_median", "below_rival", "unica", "comfyui", "torch", "cuda", "python", "gpu", "driver",
          "power_limit"]


def call(path: str, body: bytes | dict | None = None, content_type: str = "application/json"):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    request = urllib.request.Request(URL + path, data, {"Content-Type": content_type})
    with urllib.request.urlopen(request) as response:
        reply = response.read()
    return json.loads(reply) if reply else None


def upload_image(path: Path) -> None:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    call("/upload/image", body, f"multipart/form-data; boundary={boundary}")


def free() -> None:
    call("/free", {"unload_models": True, "free_memory": True})
    time.sleep(3)


def output(*command: str) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def environment(comfyui_dir: str | None) -> dict:
    """What the times depend on, as ComfyUI, the driver and git report it. No addresses or paths."""
    stats = call("/system_stats")
    system = stats["system"]
    comfyui = system["comfyui_version"]
    if comfyui_dir and (commit := output("git", "-C", comfyui_dir, "rev-parse", "--short=12", "HEAD")):
        comfyui += f" ({commit})"
    gpu, driver, power = (output("nvidia-smi", "--query-gpu=name,driver_version,power.limit",
                                 "--format=csv,noheader").split(", ") + ["", "", ""])[:3]
    release = re.search(r"release ([\d.]+)", output("nvcc", "--version"))
    return {"unica": output("git", "-C", str(REPO), "describe", "--always", "--dirty", "--tags"),
            "comfyui": comfyui, "torch": system["pytorch_version"], "cuda": release[1] if release else "",
            "python": system["python_version"].split()[0], "gpu": gpu, "driver": driver, "power_limit": power}


def graph(model: str, mode: str, measure_every: int, tag: str) -> dict:
    nodes = json.loads((GRAPHS / f"{model}.json").read_text(encoding="utf-8"))
    for name, node in list(nodes.items()):
        if node["class_type"] != "UnicaAttentionNative":
            continue
        unica_on = mode not in ("off", *SAGE)
        # the tag differs per run, so ComfyUI runs the sampler again instead of returning its cached result
        node["inputs"].update(enabled=unica_on, measure_every=measure_every, measure_tag=tag,
                              v_scale=mode if mode in RIVAL and mode != "fast" else "per_token",
                              mode="fast" if mode == "fast" else "accurate")
        if mode in SAGE:
            nodes[f"sage_{name}"] = {"class_type": "PathchSageAttentionKJ", "inputs": {
                "model": node["inputs"]["model"], "sage_attention": SAGE[mode], "allow_compile": False}}
            node["inputs"]["model"] = [f"sage_{name}", 0]
    return nodes


class Restarted(Exception):
    """ComfyUI went away during a render and came back: the models are no longer loaded."""


def wait_for_comfyui() -> None:
    deadline = time.monotonic() + RESTART_WAIT_SECONDS
    while time.monotonic() < deadline:
        time.sleep(5)
        try:
            call("/system_stats")
            return
        except OSError:
            pass
    raise SystemExit(f"ComfyUI did not come back within {RESTART_WAIT_SECONDS} s; the rows so far are in {OUT_CSV}")


def render(model: str, mode: str, measure_every: int, tag: str) -> float | None:
    """One render; its time from ComfyUI's execution_start to execution_success, or None if it failed.
    Raises Restarted if ComfyUI died during it (LOG M4.26: once in 14 days, in its dynamic VRAM, aimdo.dll)."""
    try:
        prompt_id = call("/prompt", {"prompt": graph(model, mode, measure_every, tag)})["prompt_id"]
        while True:
            history = call(f"/history/{prompt_id}").get(prompt_id)
            if history and history["status"].get("completed") is not None:
                break
            time.sleep(POLL_SECONDS)
    except urllib.error.HTTPError as error:
        print(f"  {tag}: rejected: {error.read().decode()[:2000]}")
        return None
    except OSError as error:  # connection refused or reset: the process is gone
        print(f"  {tag}: ComfyUI went away ({error}), waiting for it to restart")
        wait_for_comfyui()
        raise Restarted from error
    messages = dict(history["status"]["messages"])
    if "execution_success" not in messages:
        failure = messages.get("execution_error", {})
        print(f"  {tag}: {history['status']['status_str']}: {failure.get('node_type')} "
              f"{failure.get('exception_message')}")
        return None
    seconds = (messages["execution_success"]["timestamp"] - messages["execution_start"]["timestamp"]) / 1000
    print(f"  {tag}: {seconds:.1f} s", flush=True)
    return seconds


def measured_errors(tag: str) -> dict[str, dict]:
    """Per mode: its error on the node's measured calls of the run `tag`, and against its rival on the same calls.
    The node measures every mode on each of these calls, whichever mode the run itself has."""
    if not ERRORS_CSV.exists():
        return {}
    with ERRORS_CSV.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["tag"] == tag]
    column = {mode: [float(row[f"{name}_rel_l2"]) for row in rows if row.get(f"{name}_rel_l2")]
              for mode, name in ERROR_COLUMNS.items()}
    result = {}
    for mode, errors in column.items():
        if not errors:  # no rows, or SageAttention 3 not installed
            continue
        result[mode] = {"calls": len(errors), "rel_l2_median": statistics.median(errors), "rel_l2_max": max(errors)}
        if (rival := RIVAL.get(mode)) and len(column[rival]) == len(errors):
            result[mode] |= {"rival": rival, "rival_rel_l2_median": statistics.median(column[rival]),
                             "below_rival": sum(e < r for e, r in zip(errors, column[rival], strict=True))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("models", nargs="*", default=list(MEASURE_EVERY), choices=list(MEASURE_EVERY))
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--comfyui-dir", help="ComfyUI's checkout, to record its commit")
    args = parser.parse_args()

    env = environment(args.comfyui_dir)
    print(", ".join(f"{key} {value}" for key, value in env.items()))
    upload_image(INPUT_IMAGE)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    OUT_CSV.parent.mkdir(exist_ok=True)
    new = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()

        def record(model, mode, round_, **values):
            writer.writerow({"time": datetime.now().isoformat(timespec="seconds"), "model": model, "mode": mode,
                             "round": round_, **values, **env})
            f.flush()

        summary = {}
        for model in args.models:
            print(model)
            free()
            measured_tag = f"{model}_{stamp}_measured"
            try:
                render(model, "per_token", MEASURE_EVERY[model], measured_tag)
            except Restarted:
                render(model, "per_token", 0, f"{measured_tag}_reload")  # the models again, untimed
            errors = measured_errors(measured_tag)
            for mode, values in errors.items():
                record(model, mode, "measured", **values)
            times = {mode: [] for mode in MODES}
            for round_ in range(args.rounds):
                for mode in MODES if round_ % 2 == 0 else reversed(MODES):
                    tag = f"{model}_{stamp}_r{round_}_{mode}"
                    try:
                        seconds = render(model, mode, 0, tag)
                    except Restarted:  # this render has no time; load the models again so the next one starts warm
                        seconds = None
                        render(model, "per_token", 0, f"{tag}_reload")
                    record(model, mode, round_, seconds=seconds)
                    if seconds is not None:
                        times[mode].append(seconds)
            summary[model] = times, errors
        free()

    print(f"\nmedian of {args.rounds} renders; error: median (max) rel L2 over the measured calls")
    for model, (times, errors) in summary.items():
        off = statistics.median(times["off"]) if times["off"] else None
        print(model)
        for mode in MODES:
            if not times[mode]:
                print(f"  {mode:12} failed")
                continue
            seconds = statistics.median(times[mode])
            line = f"  {mode:12} {seconds:7.1f} s"
            if off and mode != "off":
                line += f" ({seconds / off - 1:+6.1%} vs off)"
            if mode in errors:
                e = errors[mode]
                line += f"   error {100 * e['rel_l2_median']:.2f} % ({100 * e['rel_l2_max']:.2f})"
                if "rival" in e:
                    line += f", below {e['rival']} in {e['below_rival']}/{e['calls']} calls"
            print(line)


if __name__ == "__main__":
    main()
