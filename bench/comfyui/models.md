# The models of the bench graphs

The five graphs of [comfyui_models.py](../comfyui_models.py) need these files. Paths are relative to `ComfyUI/models`.
A `?` means the source is not recorded yet.

| graph | folder | file | bytes | URL |
|---|---|---|---|---|
| wan | diffusion_models | wan2.2_i2v_high_noise_14B_nvfp4_mixed.safetensors | 10535921496 | ? |
| wan | diffusion_models | wan2.2_i2v_low_noise_14B_nvfp4_mixed.safetensors | 10535921496 | ? |
| wan | loras | wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors | 1226977424 | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors |
| wan | loras | wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors | 1226977424 | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors |
| wan | text_encoders | umt5_xxl_fp8_e4m3fn_scaled.safetensors | 6735906897 | https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors |
| wan | vae | Wan2.1_VAE.pth | 507609880 | ? (the .pth variant; Comfy-Org ships wan_2.1_vae.safetensors) |
| ltx23 | diffusion_models | ltx-2.3-22b-dev-nvfp4.safetensors | 21701367310 | ? |
| ltx23 | loras | ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors | 2741024390 | ? |
| ltx23 | clip | gemma_3_12B_it_fp4_mixed.safetensors | 9447702218 | https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors |
| ltx23 | clip | ltx-2.3_text_projection_bf16.safetensors | 2312149072 | ? |
| ltx23 | latent_upscale_models | ltx-2.3-spatial-upscaler-x2-1.1.safetensors | 995743560 | https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors |
| ltx23 | vae | LTX23_video_vae_bf16.safetensors | 1452258578 | ? |
| ltx23 | vae | LTX23_audio_vae_bf16.safetensors | 364855188 | ? |
| qwen_edit | diffusion_models | qwen_image_edit_2511_int8_convrot.safetensors | 20499083824 | ? |
| qwen_edit | loras | Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors | 849608296 | ? |
| qwen_edit | text_encoders | qwen_2.5_vl_7b_fp8_scaled.safetensors | 9384670680 | https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors |
| qwen_edit | vae | qwen_image_vae.safetensors | 253806246 | https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors |
| zimage | diffusion_models/zimage | z_image_int8_convrot.safetensors | 6201001296 | ? |
| zimage | clip | qwen_3_4b.safetensors | 8044982048 | https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors |
| zimage | vae | ae.safetensors | 335304388 | https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/resolve/main/split_files/vae/ae.safetensors |
| moss | diffusion_models/moss | miaomiaoHaremMOSS_baseV10_int8.safetensors | 5413700574 | ? (Civitai merge, not on HF) |
| moss | text_encoders | qwen3.5_9b_w4a8_gptq.safetensors | 7343591088 | ? |
| moss | vae | flux2-vae_small.safetensors | 249521340 | ? |

Besides the models:

- **Custom nodes.** The graphs need:
  - the Unica node, installed from this repository: link `custom_nodes/UnicaAttention` to `comfyui/`. The script
    then reads the errors the node writes to `build/`;
  - KJNodes (1.3.9) with SageAttention 2 (2.2.0) and SageAttention 3 (1.0.0), for the sage2 and sage3 modes.
- **Input image.** [unica_bench.png](unica_bench.png) is used by wan and qwen_edit. The script uploads it itself.

## What differs from the studio workflows

The graphs come from the workflows the earlier measurements ran (LOG M4.22–M4.25). Whatever the attention does not
touch is removed from them:

- **wan:** no SageAttention 3 patch (KJNodes) under the Unica node, so calls the node passes on go to ComfyUI's own
  attention; no RIFE frame interpolation (it was switched off).
- **ltx23:** the prompt goes to the text encoder as written. Gemma no longer rewrites it first, which cost time and
  could change from run to run.
- **moss:** no 2x upscaler after decoding (it was switched off).
- **all:** the input image is downscaled 2x to 1080 x 1920, without the workflow ComfyUI had saved into it.

## The environment of the first run (2026-11-05)

ComfyUI master 067f853305e7 (2026-10-24), run without extra flags; Python 3.12.10, torch 2.14.0+cu132, CUDA 13.2,
driver 617.14, RTX 5080 at a 360 W power limit. The script writes the environment into every row of its CSV.
