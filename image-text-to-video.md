# Image-Text-to-Video (`image-text-to-video`)

## Overview
Image-Text-to-Video models take a static image as an initial anchor frame and generate a coherent, high-definition video sequence guided by textual motion prompts. The model must infer realistic temporal dynamics, camera movements, and physical interactions that naturally extend the starting visual scene. By combining spatial image representations with temporal attention layers or spatiotemporal diffusion transformers, these architectures synthesize smooth frame transitions while maintaining visual fidelity and subject consistency over time. This capability is pivotal for automated cinematography, animated advertising, video game cinematic production, digital marketing campaigns, and converting static artwork or photographs into dynamic visual media.

## Representative Model: `MiniMaxAI/MiniMax-H3`

### Model profile
MiniMax H3 is a synchronized audio-video generation system, not just an image animator. It combines a 33.1B omni transformer with a 32B Qwen3-VL-32B text encoder and visual/audio VAEs; the repository lists roughly 498 GB of files (355 GB used storage). It produces 4–15 s clips at 24 fps and 768p locally, with 2K through a hosted regeneration stage, and dialogue in 11 languages. The license is the custom MiniMax H3 Community License.

### Architecture tailored to image-text-to-video
A single-stream dense Omni Transformer runs over packed multimodal tokens with 3D MM-RoPE and modality-specific AdaLN modulation. The H3-Encoder conditions on layer-50 hidden states from Qwen3-VL-32B; the visual VAE uses 1×2×2 patching (32× spatial, 4× temporal) and the audio VAE runs at 40 Hz/channel. A single denoiser therefore coordinates appearance, motion, and sound. Shared theory is in [diffusion and flow matching](roadmap.md#diffusion-flow-matching-and-vaes).

### Niche mechanism
H3 packs text, image, video, and audio tokens into one sequence and runs a single dense transformer with 3D MM-RoPE and modality-specific AdaLN, so appearance, motion, and sound share one denoiser. The visual VAE uses 1×2×2 patching (32× spatial, 4× temporal) and a 40 Hz audio VAE; Qwen3-VL layer-50 features provide semantics. **Design takeaway:** one packed token stream + modality-specific modulation is simpler than separate cross-attention towers, but it needs a strong shared encoder and VAEs.

### Training and task tailoring
The card/blog does not disclose data or objectives beyond native sparse attention in the final training stage. The open release covers H3-Base for FL2VA and Ref2VA, is CFG-distilled, and omits H3-Context-IR, H3-Regenerate-2K, and sparse attention, which remain API-only.

### How it is used
Use Diffusers' ModularPipeline, SGLang, vLLM, or ComfyUI in BF16; the example uses four GPUs. Inputs can include up to 9 images and 3 video/audio clips (12 files total). Local output is 768p; full 2K requires the hosted pipeline.

### Fit and caveats
The tag only partly fits: this is a full audio-video system with reference conditioning, not a simple image-to-video model. There are no numeric benchmarks on the card, the open weights omit the 2K components, and the 33B+32B stack is expensive to run.

## Related Reading
- [Roadmap: diffusion and flow matching](roadmap.md#diffusion-flow-matching-and-vaes) and [MiniMax H3 family](roadmap.md#model-family-notes).
- [Text-to-Video](text-to-video.md) · [Image-to-Video](image-to-video.md) · [Any-to-Any](any-to-any.md)
