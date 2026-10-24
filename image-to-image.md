# Image-to-Image (`image-to-image`)

## Overview
Image-to-Image is a versatile computer vision paradigm that transforms an input source image into an altered target image while preserving fundamental spatial structure and semantic content. This encompasses a broad spectrum of translation tasks including super-resolution upscaling, sketch-to-photo synthesis, grayscale colorization, semantic map rendering, and artifact inpainting. The underlying networks leverage conditional generative adversarial networks or diffusion models guided by edge maps or latent representations. Image-to-Image models are widely employed in film remastering, architectural rendering, medical scan enhancement, digital photo restoration, and creative digital art production.

## Representative Model: `Qwen/Qwen-Image-Edit-2511`

### Model profile
Qwen-Image-Edit-2511 is an instruction-based image editor released in December 2025 under Apache-2.0. It combines a 20.43B MMDiT denoiser with an 8.29B Qwen2.5-VL text encoder (about 57.7 GB BF16). It accepts one or more images plus a text instruction and returns an edited image; it is not a classic SD-style img2img pipeline.

### Architecture tailored to instruction editing
The `QwenImageTransformer2DModel` MMDiT has 60 layers, 24 heads, patch 2, 64 input/16 output latent channels, and joint attention width 3584. A `AutoencoderKLQwenImage` VAE (z_dim 16) handles pixels, and the Qwen2.5-VL encoder supplies rich instruction and reference-image features. Joint text-image attention is what lets the model preserve subject identity while applying local edits. See [MMDiT family](roadmap.md#model-family-notes).

### Niche mechanism
MMDiT keeps text and image tokens in separate streams but lets them attend jointly, so an instruction can modify a local region while the source-image latent anchors identity. The denoiser works in a 16-channel latent space with patch 2 and 3584-wide joint attention; a flow-matching Euler sampler moves from noise to the edited latent. **Design takeaway:** for editing, condition the denoiser on the source latent, not just a text prompt; joint attention preserves structure.

### Training and task tailoring
The 2511 card only says it improves on 2509 (September 2025): less image drift, better character and multi-person consistency, built-in community LoRAs, and stronger industrial-design and geometric reasoning. The 2511-specific data and objectives are not stated; the base Qwen-Image report describes flow-matching MMDiT pretraining plus RL post-training. Do not transfer base-model benchmarks to this checkpoint.

### How it is used
Use the latest Diffusers `QwenImageEditPlusPipeline` in BF16 with `image=[img1, img2]` and a prompt. Typical settings are 40 steps, `true_cfg_scale=4.0`, `guidance_scale=1.0`, and a single-space negative prompt.

### Fit and caveats
The tag understates what this does: it is an instruction/multi-image editor with about 28.7B total weights, so it needs a large GPU and recent Diffusers. There are no official 2511 benchmark numbers; third-party CPI-Bench reports GEdit 7.87, ImgEdit 4.51, and REDEdit 4.23.

## Related Reading
- [Roadmap: MMDiT family](roadmap.md#model-family-notes) and [diffusion and flow matching](roadmap.md#diffusion-flow-matching-and-vaes).
- [Image-Text-to-Image](image-text-to-image.md) · [Text-to-Image](text-to-image.md) · [Image-to-Video](image-to-video.md)
