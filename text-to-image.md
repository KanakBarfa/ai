# Text-to-Image (`text-to-image`)

## Overview
Text-to-Image models synthesize high-resolution, visually compelling digital images directly from written natural language descriptions. Modern architectures rely on diffusion models, flow matching mechanisms, and multi-modal diffusion transformers conditioned on rich textual embeddings from large language models. The network iteratively refines random Gaussian noise into coherent visual structures, adhering closely to the artistic style, spatial layout, lighting, and semantic details outlined in the text prompt. This capability has revolutionized digital art, concept design, advertising, book illustration, interior styling, and creative prototyping across global media industries.

## Representative Model: `krea/Krea-2-Turbo`

### Model profile
Krea 2 Turbo is a 12.82B-parameter text-to-image model (12.50B BF16 plus 0.32B F32) released in mid-2026 under the Krea 2 Community License. The Hub card is gated and largely empty, so most details come from Krea's technical report. It targets roughly 1K–2K images from English prompts and is available through Diffusers as `Krea2Pipeline`.

### Architecture tailored to prompt adherence
It is a single-stream MMDiT with grouped-query attention using sigmoid-gated attention, SwiGLU, zero-centered RMSNorm plus QKNorm, 3D axial RoPE, and a per-block bias for timestep modulation. Text conditioning uses Qwen3-VL with layerwise feature aggregation, and the VAE is FLUX.2's. The LLM-grade text encoder and layerwise aggregation are the main reasons it can follow long, compositional prompts. See [MMDiT family](roadmap.md#model-family-notes).

### Niche mechanism
Krea 2 is a single-stream MMDiT: text and image tokens share one attention stream, with sigmoid-gated GQA, QK-norm, zero-centered RMSNorm, 3D axial RoPE, and per-block timestep bias. Rectified-flow v-prediction is followed by STPO and GRPO, then TDM distillation to 8 steps. **Design takeaway:** prompt adherence comes from a strong layerwise text encoder plus preference/RL, while QK-norm and 3D RoPE stabilize the denoiser.

### Training and task tailoring
Training starts with rectified-flow pretraining and v-parameterization at 256→512→1024 px, including 8-bit low-resolution training. It then goes through midtraining, aesthetic SFT, two-stage preference optimization (STPO), and GRPO-style RL with aesthetic, prompt-following, text-rendering, and artifact rewards. Turbo adds guidance and timestep distillation via Trajectory Distribution Matching. Pretraining uses billions of curated images with no AI-generated samples.

### How it is used
Use Diffusers with 8 steps, CFG 0, shift mu 1.15, and 1K–2K output in BF16.

### Fit and caveats
The tag fits, but the checkpoint is 12.8B, the license is custom, and the gated/empty card prevents independent verification. Krea's report claims a top-10 Artificial Analysis leaderboard position and second among independent labs, but gives no GenEval/DPG table. Treat the ranking as vendor-reported.

## Related Reading
- [Roadmap: MMDiT family](roadmap.md#model-family-notes) and [diffusion and flow matching](roadmap.md#diffusion-flow-matching-and-vaes).
- [Image-to-Image](image-to-image.md) · [Image-Text-to-Image](image-text-to-image.md) · [Unconditional Image Generation](unconditional-image-generation.md)
