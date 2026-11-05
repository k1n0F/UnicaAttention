"""The bench graphs of bench/comfyui_models.py: every graph has its Unica nodes, and graph() changes only them, plus a
SageAttention patch before each of them in the sage modes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import comfyui_models


def test_graphs_change_only_the_unica_nodes():
    for model in comfyui_models.MEASURE_EVERY:
        plain = comfyui_models.graph(model, "per_token", 0, "x")
        unica = {name for name, node in plain.items() if node["class_type"] == "UnicaAttentionNative"}
        assert unica, model
        for mode in comfyui_models.MODES:
            nodes = comfyui_models.graph(model, mode, 5, f"{model}_{mode}")
            sage = {f"sage_{name}" for name in unica} if mode in comfyui_models.SAGE else set()
            assert nodes.keys() == plain.keys() | sage
            assert all(nodes[n] == plain[n] for n in plain.keys() - unica)
            for name in unica:
                inputs = nodes[name]["inputs"]
                assert inputs["enabled"] == (mode not in ("off", *comfyui_models.SAGE))
                assert inputs["measure_every"] == 5 and inputs["measure_tag"] == f"{model}_{mode}"
                assert inputs["mode"] == ("fast" if mode == "fast" else "accurate")
                assert inputs["v_scale"] == (mode if mode in ("per_channel", "per_token", "outliers") else "per_token")
                if sage:  # the patch sits between the model and the disabled Unica node
                    assert inputs["model"] == [f"sage_{name}", 0]
                    assert nodes[f"sage_{name}"]["inputs"]["model"] == plain[name]["inputs"]["model"]
            images = [x["inputs"]["image"] for x in nodes.values() if x["class_type"] == "LoadImage"]
            assert all(image == comfyui_models.INPUT_IMAGE.name for image in images)
