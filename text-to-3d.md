# Text-to-3D (`text-to-3d`)

## Overview
Text-to-3D models generate fully realized three-dimensional digital assets from written descriptive prompts. These architectures utilize techniques such as score distillation sampling from two-dimensional diffusion priors, direct three-dimensional native diffusion, or autoregressive generation of polygon mesh tokens. The resulting outputs include textured polygon meshes, neural radiance fields, or three-dimensional Gaussian splats that can be rotated, illuminated, and inspected from any viewing perspective. Text-to-3D technology drastically accelerates asset creation workflows in video game development, visual effects production, virtual reality environment design, and industrial prototyping.

## Representative Model: `Zhengyi/LLaMA-Mesh`

### Model profile
LLaMA-Mesh is an 8.03B-parameter model (BF16, 16 GB) released in November 2024 under NSCLv1 (non-commercial) plus the Llama 3.1 Community License. It takes a text prompt and emits OBJ-style plain text with vertex (`v`) and face (`f`) lines, which the user parses into a mesh. It is a language model that writes mesh files, not a native 3D generator.

### Architecture tailored to mesh-as-text
The base is Llama 3.1 8B: a 32-layer decoder with 4096 hidden, 32 attention/8 KV heads, and a 128,256-token vocabulary. There is no 3D tokenizer, voxel grid, or diffusion component. The task adaptation is entirely representational: vertices and faces are tokenized as ordinary text, so the model can reuse language pretraining and emit a mesh autoregressively. See [text generation](text-generation.md).

### Niche mechanism
There is no 3D architecture: OBJ vertices and faces are serialized as text and predicted with the same next-token objective as language. The 500-face cap keeps the sequence within the 8K-token budget; the model reuses Llama's text prior at the cost of geometric detail and validity. **Design takeaway:** serialization is a valid prototype, but a native sparse/voxel or mesh-token representation scales better.

### Training and task tailoring
Training is supervised fine-tuning only, starting from Llama 3.1 8B. The 3D corpus is 30K Objaverse meshes filtered to fewer than 500 faces, mixed with UltraChat instructions to preserve language ability. It used 32 GPUs and 2,400 GPU-hours, versus 6,000 for MeshXL. The face cap keeps sequences within the context window.

### How it is used
Use Transformers, prompt for an OBJ mesh, generate up to about 8K tokens, and parse the text into a `.obj` file. Output validity is not guaranteed; malformed lines must be handled.

### Fit and caveats
The tag fits, but the model emits text rather than a mesh-native representation, detail is capped at about 500 faces, and invalid OBJ output is possible. MMLU stays at 66.07, while GSM8K drops from 77.18 to 62.09, showing some language regression. Mesh quality is reported as comparable to MeshXL/Unique3D qualitatively. The non-commercial license limits production use.

## Related Reading
- [Roadmap: masked vs. autoregressive objectives](roadmap.md#masked-vs-autoregressive-objectives) and [transformers](roadmap.md#transformers-and-attention).
- [Text Generation](text-generation.md) · [Image-to-3D](image-to-3d.md) · [Image-to-Image](image-to-image.md)
