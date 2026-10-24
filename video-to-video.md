# Video-to-Video (`video-to-video`)

## Overview
Video-to-Video models transform an existing video sequence into a newly stylized or modified video output while maintaining the original motion paths, camera dynamics, and timing. Applications include video style transfer, temporal super-resolution, background replacement, character reskinning, and frame-rate interpolation. These architectures incorporate inter-frame cross-attention and optical flow warping constraints to ensure that modifications do not produce temporal flickering or inconsistent structural artifacts between consecutive frames. Video-to-Video tools are essential in visual effects post-production, animation production pipelines, and automated film remastering.

## Representative Model: `Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen`

### Model profile
Alpha-Gen is a reference-conditioned IC-LoRA adapter (about 1.33 GB repository) for LTX-2.5-22B, released in October 2026 under the gated LTX-2.x Community License. It converts an RGB "beauty" video into a grayscale alpha matte (white foreground, black background). It does not output RGB video, and it takes no prompt or mask.

### Architecture tailored to video matting
The adapter is an IC-LoRA on the LTX-2.5-22B DiT. The RGB clip is VAE-encoded and kept attached throughout denoising, so the predicted matte stays spatially and temporally aligned with the source; inference runs stage 1 only with an empty prompt. This reference-conditioning path is what preserves hair, fur, smoke, fire, sheer fabric, glass, and water edges across frames. See [LTX family](roadmap.md#model-family-notes).

### Niche mechanism
IC-LoRA keeps the source video's VAE latents attached through every denoising step, so the generated alpha matte is conditioned on the actual RGB frames rather than a text prompt. The adapter is reference-conditioned, the prompt is empty, and stage 2 is skipped; the output is a matte, not RGB. **Design takeaway:** reference-conditioned video-to-video needs the reference latents in the denoiser; an empty prompt is fine when the signal is visual.

### Training and task tailoring
Training ran 25K steps of IC-LoRA. The card does not state the dataset composition or objectives beyond reference-conditioned video-to-video, and no RL stage is reported. The qualitative examples target difficult transparency and motion-blur cases.

### How it is used
Run `python -m ltx_pipelines.ic_lora` with the distilled pipeline, `--lora ... 1.0`, `--video-conditioning input.mp4 1.0`, an empty `--prompt ""`, and `--skip-stage-2`. Set width/height to twice the source (3840×2176 for a 1920×1088 output). Limits are 145 frames and 1920×1088.

### Fit and caveats
The tag is video-to-video, but the output is an alpha matte that must be composited with the original clip. There is no prompt/mask control over which object is foreground, RGB can leak into the matte beyond 145 frames, and full-HD clips need an H100/B200-class GPU. No quantitative matting benchmark is published.

## Related Reading
- [Roadmap: LTX family](roadmap.md#model-family-notes) and [adapters, LoRA, and quantization](roadmap.md#adapters-lora-and-quantization).
- [Image-to-Video](image-to-video.md) · [Video Classification](video-classification.md) · [Image Segmentation](image-segmentation.md)
