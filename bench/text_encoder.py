"""M4.36: the text encoder of "wan stable work" (umt5-xxl), as ComfyUI runs it: comfy.sd.load_clip and
encode_from_tokens_scheduled, the prompt and the negative read from the workflow. Run it with ComfyUI's Python and the
thread variables ComfyUI is launched with (OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=8):

    <ComfyUI>/.venv-13/Scripts/python.exe bench/text_encoder.py <ComfyUI dir> <workflow .json> [cuda]

Each prompt is padded to 512 tokens. On the CPU ComfyUI keeps the weights in the file's dtype (fp16; the dtype option is
overridden by the one detected in the file) and casts every Linear's weight to fp32 on every call. "fp32 weights"
converts them once after loading, which separates the cast from the matmuls. Each configuration encodes both prompts
REPEATS times; on the GPU the first time includes uploading the weights. The error is the conditioning's rel L2
against the render's path (CPU, fp16 weights).
"""

import json
import sys
import time
from pathlib import Path

import torch

COMFY = Path(sys.argv[1])
WORKFLOW = Path(sys.argv[2])
WITH_CUDA = len(sys.argv) > 3 and sys.argv[3] == "cuda"
sys.path.insert(0, str(COMFY))

import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402

ENCODER = COMFY / "models" / "clip" / "umt5_xxl_fp16.safetensors"
PROMPT_NODES = (88, 86)  # the workflow's positive and negative CLIPTextEncode
REPEATS = 2
CPU = torch.device("cpu")


def read_prompts() -> list[str]:
    nodes = {node["id"]: node for node in json.loads(WORKFLOW.read_text(encoding="utf-8"))["nodes"]}
    return [nodes[node_id]["widgets_values"][0] for node_id in PROMPT_NODES]


def load(device: torch.device, weights: torch.dtype):
    start = time.perf_counter()
    clip = comfy.sd.load_clip(ckpt_paths=[str(ENCODER)], clip_type=comfy.sd.CLIPType.WAN,
                              model_options={"load_device": device, "offload_device": CPU})
    if weights == torch.float32:
        clip.cond_stage_model.float()
    return clip, time.perf_counter() - start


def encode(clip, prompts: list[str]) -> tuple[list[torch.Tensor], float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    conditionings = [clip.encode_from_tokens_scheduled(clip.tokenize(text))[0][0].float().cpu() for text in prompts]
    return conditionings, time.perf_counter() - start


def relative_l2(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max(((x - y).norm() / y.norm()).item() for x, y in zip(a, b))


def main():
    prompts = read_prompts()
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, prompts {[len(p) for p in prompts]} chars")
    configurations = [(CPU, torch.float16, 8), (CPU, torch.float16, 16), (CPU, torch.float32, 8),
                      (CPU, torch.float32, 16)]
    if WITH_CUDA:
        configurations.append((torch.device("cuda"), torch.float16, 8))  # threads don't matter on the GPU
    reference, loaded = None, None
    for device, dtype, threads in configurations:
        if loaded is None or loaded[1:] != (device, dtype):
            loaded = None
            comfy.model_management.unload_all_models()
            clip, load_seconds = load(device, dtype)
            loaded = (clip, device, dtype)
            print(f"load {device} {str(dtype)[6:]} weights: {load_seconds:.1f} s")
        torch.set_num_threads(threads)
        times = []
        for _ in range(REPEATS):
            conditionings, seconds = encode(loaded[0], prompts)
            times.append(seconds)
        reference = reference or conditionings
        error = relative_l2(conditionings, reference)
        print(f"{device.type:4} {str(dtype)[6:]:7} weights {threads:2} threads: "
              + ", ".join(f"{seconds:.2f}" for seconds in times) + f" s, rel L2 vs cpu fp16 {error * 100:.3f} %",
              flush=True)


if __name__ == "__main__":
    main()
