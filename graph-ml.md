# Graph Machine Learning (`graph-ml`)

## Overview
Graph Machine Learning models analyze relational graph structures composed of interconnected nodes and connecting edges. Utilizing Graph Neural Networks, Graph Attention Networks, or Graph Transformers, these models pass message representations across local graph neighborhoods to capture topology and relational dependencies. Key tasks include node classification, edge prediction, and whole-graph property regression. By modeling non-Euclidean relationships, these models solve problems flat vectors cannot represent. Graph ML is indispensable for predicting molecular properties in drug discovery, detecting financial fraud rings in transaction graphs, recommendation networks in social media, and knowledge graph completion.

## Representative Model: `Disya/UIGEN-T3-14B-Preview-Q4_K_M-GGUF`

### Model profile
This is a Q4_K_M GGUF conversion of UIGEN-T3-14B-Preview, storing 14.77B parameters in 9.0 GB (~4.9 bits/weight). It was created in June 2025. The upstream model is research-only/non-commercial, but this card omits the license. Despite the `graph-ml` tag, it generates HTML/CSS/Tailwind UI code from text prompts.

### Architecture and task reality
The base is Qwen3-14B: a 40-layer dense decoder with 5120 hidden, 40 Q/8 KV heads, head_dim 128, FFN 17,408, a 151,936 vocabulary, and 40,960 positions. Upstream metadata suggests a PEFT/LoRA fine-tune, and the matching parameter count implies a merged adapter. It supports `/think` and `/no_think` and is trained to emit design tokens before code. There is no graph encoder, message passing, or node/edge attention.

### Niche mechanism
There is no graph mechanism: Qwen3-14B is a 40L dense causal decoder with standard 1-D RoPE. It cannot see edges, node identities, or permutation structure; any graph must be serialized into text first. Real graph learning needs message passing or a relational attention bias that is invariant to node order. **Design takeaway:** a graph tag on an LLM is not a graph model; serialize only if you accept the loss of topology.

### Training and task tailoring
It is fine-tuned from Qwen3-14B for UI/HTML/Tailwind generation with a pre/post reasoning format. Dataset size and loss are not stated. GRPO is roadmap-only, and UIGenEval is intended to inform future rewards. The task adaptation is therefore SFT on UI code, not graph learning.

### How it is used
Run with llama.cpp `llama-cli` or `llama-server` using the repository file. Recommended settings are temperature 0.6, top_p 0.95, top_k 20, `/think` or `/no_think`, and about 20K context for reasoning (40K+ max tokens). The card's 2,048-token example is far below that and will truncate reasoning.

### Fit and caveats
The tag is wrong: this is UI code generation, not graph machine learning. The GGUF is a third-party conversion that omits the upstream non-commercial research-only license. Q4_K_M is lossy, and neither card reports quantitative results—UIGenEval is announced but has no scores.

## Related Reading
- [Roadmap: task-tag caveats](roadmap.md#task-tag-caveats) and [adapters, LoRA, and quantization](roadmap.md#adapters-lora-and-quantization).
- [Text Generation](text-generation.md) · [Image-to-Text](image-to-text.md) · [Summarization](summarization.md)
