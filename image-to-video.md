# Image-to-Video (`image-to-video`)

## Overview
Image-to-Video models take a single static reference image and generate plausible cinematic motion extending from it, guided by optional motion parameters or text prompts. The architecture treats the reference image as the initial conditioning anchor, synthesizing subsequent frames that adhere to the original lighting, texture, and character proportions while introducing natural movement. This technology brings static historical portraits to life, animates digital concept art for film pitches, generates looping cinemagraphs for digital marketing, and streamlines the pre-visualization workflow for directors and visual artists.

## Representative Model: `Lightricks/LTX-2.5-22b-LoRA-Cinemagraph`

### Model profile
This is a roughly 207 MB LoRA adapter (`.safetensors`), released in October 2026 under the gated LTX-2.x Community License. It is not a standalone model: it adapts the 22B LTX-2.5 image-to-video backbone to produce looping cinemagraphs from a still image plus a text prompt.

### Architecture tailored to cinemagraphs
The adapter is a LoRA fine-tune of the LTX-2.5-22B DiT, used with the distilled 8-step checkpoint. The base combines a Gemma 4 12B text encoder with video and audio VAEs. The LoRA adds selective-motion behavior without a separate motion-control network, so the prompt token `CINEMAGRAPH_MOTION` and adapter strength steer which element moves. See [LTX family](roadmap.md#model-family-notes).

### Niche mechanism
The adapter is a rank-32 LoRA on the LTX-2.5 DiT's projections, trained for 1,000 steps on locked-off tripod clips with one moving element. The distilled base already samples in 8 steps; the LoRA only shifts the motion prior toward looping, low-amplitude movement, and the `CINEMAGRAPH_MOTION` token selects that regime. **Design takeaway:** a small adapter can specialize a general video DiT; data selection defines the motion grammar more than architecture does.

### Training and task tailoring
Training uses LoRA rank 32, alpha 32, learning rate 1e-4, batch 1, AdamW-8bit, cosine schedule, and 1,000 steps. The dataset is "Cinemagraph (No Moving People)": locked-off tripod shots with a single moving element such as water, neon, leaves, or reflections. No RL is reported. The narrow dataset is why the adapter should not be expected to animate people or move the camera.

### How it is used
Run `ltx_pipelines.distilled` or ComfyUI T2V/I2V with the LoRA at strength 1.0–1.2. Include `CINEMAGRAPH_MOTION` in the prompt. Recommended settings are 30 steps, CFG 4.0, STG `stg_v` scale 1.0 on block 29, 512×704, and 25 frames.

### Fit and caveats
The tag fits, but the adapter is deliberately narrow: static camera, no body movement, one moving element. There is no quantitative benchmark, and the base model is gated. Treat it as a creative effect, not a general image animator.

## Related Reading
- [Roadmap: LTX family](roadmap.md#model-family-notes) and [adapters, LoRA, and quantization](roadmap.md#adapters-lora-and-quantization).
- [Text-to-Video](text-to-video.md) · [Video-to-Video](video-to-video.md) · [Feature Extraction](feature-extraction.md)
