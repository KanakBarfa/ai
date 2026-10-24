# Image-Text-to-Image (`image-text-to-image`)

## Overview
Image-Text-to-Image models perform controlled image editing and synthesis by accepting an existing source image alongside explicit text instructions describing desired modifications. Rather than generating a picture from scratch, the architecture preserves the fundamental identity, pose, and structure of the original subject while applying fine-grained adjustments such as style transfer, object addition, background replacement, or lighting changes. These models use cross-attention conditioning and latent diffusion techniques to map textual edits directly to pixel modifications without introducing unintended artifacts. Common applications include commercial graphic design software, virtual try-on platforms, game asset iteration, and creative photo editing suites.

## Representative Model: `OS-Software/Qwen3.8-27B-Uncensored-Heretic-v3-GGUF`

### Model profile
This is a community derivative of Qwen3.8-27B, quantized to GGUF and modified with Heretic v2 abliteration. It stores about 26.9B parameters in 12 quantizations from IQ2_S (8.37 GB) to Q8_K_XL (30.89 GB), plus BF16/Q8 mmproj files for vision. It is Apache-2.0 and was updated in September 2026.

### Architecture and task reality
The base is a Qwen3.5/3.8-family hybrid VLM: 64 layers, 16 blocks of three Gated DeltaNet layers plus one Gated Attention layer, 5120 hidden, 24 Q/4 KV attention heads, and a 27-layer/1152-hidden ViT with patch 16 and merge 2. It accepts images/video and **returns text**. Despite the `image-text-to-image` tag, it does not generate or edit images. See [Qwen family](roadmap.md#model-family-notes).

### Niche mechanism
The tag is wrong: this is a Qwen3.5-family VLM with three Gated DeltaNet layers per Gated Attention layer, and its output is text. Abliteration edits layers 27–44 by learning low-rank directions on `attn.o_proj` and `mlp.down_proj` and subtracting them with a steer/overcorrect rule; it removes refusal behavior but adds no image decoder. **Design takeaway:** safety alignment is a separable direction in weight space; image generation requires a diffusion/VAE head, not an LLM edit.

### Training and task tailoring
Base pretraining/post-training is not documented in this derivative card. The visible tailoring is abliteration: LoRA rank 128 on `attn.o_proj` and `mlp.down_proj`, layers 27–44, steer 0.03, overcorrect 2.0, transport rank 4. The card reports 0/100 refusals versus 99/100 before, with KL divergence 0.0087. No vision or language benchmark was rerun, so base-model scores do not transfer.

### How it is used
Run with llama.cpp and the matching mmproj for vision. Q4_K_M is 16.1 GB; IQ2_S is 8.4 GB. Context can reach 262K, but memory scales accordingly. Thinking is on by default, controlled by `reasoning_effort` (xhigh/medium/low); use the Qwen chat template and pass images as media.

### Fit and caveats
This is a major tag mismatch: it is a text-output VLM, not an image generator. Safety alignment is deliberately removed, so outputs may be harmful. Low-bit quantization adds quality loss, vision requires mmproj, and the 27B model is heavy for CPU.

## Related Reading
- [Roadmap: task-tag caveats](roadmap.md#task-tag-caveats) and [Qwen family](roadmap.md#model-family-notes).
- [Image-Text-to-Text](image-text-to-text.md) · [Image-to-Image](image-to-image.md) · [Text-to-Image](text-to-image.md)
