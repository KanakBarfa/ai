# Text-to-Video (`text-to-video`)

## Overview
Text-to-Video models synthesize multi-frame animated video sequences from natural language descriptive prompts. Building on spatiotemporal diffusion transformers and continuous 3D latent spaces, these architectures generate consistent visual subjects, natural physical dynamics, and smooth camera trajectories across consecutive frames. The network must maintain temporal coherence, preventing flickering, morphing, and physical impossibilities over extended durations. Modern implementations handle lighting, complex character actions, and environmental interactions. Text-to-video represents a revolutionary leap forward in automated video editing, advertising creation, concept film prototyping, social media marketing, and dynamic educational content generation.

## Representative Model: `FastVideo/FastVideo-FastH3-8-Step-V2`

### Model profile
FastH3-8-Step-V2 is a 35.05B-parameter distilled video generator (BF16, about 147.8 GB) released in September 2026 under the MiniMax H3 Community License. It takes a text prompt and produces video **plus 32 kHz audio**, not silent video. It is a text-to-audio-video (T2AV) model built from MiniMax H3.

### Architecture tailored to few-step generation
It distills MiniMax H3's 33B dense single-stream omni transformer (50 blocks). It reuses H3's Qwen3-VL text encoder (layer-50 states), the 16×/4×/24-channel visual VAE, and the 40 Hz audio VAE. Generation uses nine sigma-grid points, i.e. eight forward passes. The student uses VSA-H3 attention with 80% sparsity in 64-token tiles, a key efficiency change. See [diffusion and flow matching](roadmap.md#diffusion-flow-matching-and-vaes).

### Niche mechanism
FastH3 distills the 33B H3 single-stream omni transformer into 8 forward passes with DMD2 and carried backward simulation, using student-generated latents instead of target videos. VSA-H3 attention is 80% sparse in 64-token tiles, cutting the cost of each pass; video and audio latents are denoised together. **Design takeaway:** few-step video generation = distillation + sparse attention + a joint audio-video latent space.

### Training and task tailoring
Distillation is data-free DMD2 with carried backward simulation: prompts plus student-generated latents are used instead of target videos. The checkpoint is step 1300, with video/audio shifts of 10/3 (not the base's 12/3) and guidance 1.0. Only T2AV was distilled; FL2VA and Ref2VA were not.

### How it is used
Run the FastVideo example `examples/inference/basic/basic_fasth3_8step.py`. It was tested on 4×B200 at 1344×768, 124 frames, and 24 FPS; dimensions must be multiples of 32. The VSA-H3 backend/kernel is required.

### Fit and caveats
The tag omits the audio output. It is not a stock Diffusers pipeline, and difficult motion or fine detail may remain below the base H3. One measured 4×GB200 request at 832×480 with 124 frames and audio took 6.3 s. FastVideo's blog claims V2 beats base H3 on quality, but no formal matched-quality benchmark is published.

## Related Reading
- [Roadmap: diffusion and flow matching](roadmap.md#diffusion-flow-matching-and-vaes) and [MiniMax H3 family](roadmap.md#model-family-notes).
- [Image-to-Video](image-to-video.md) · [Image-Text-to-Video](image-text-to-video.md) · [Video-to-Video](video-to-video.md)
