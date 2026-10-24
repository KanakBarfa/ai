# Feature Extraction (`feature-extraction`)

## Overview
Feature Extraction maps input text into dense, continuous numerical vector embeddings that encapsulate lexical, semantic, and contextual meaning. Pretrained on masked language modeling or contrastive objectives, these models project texts with similar conceptual meaning close to each other in vector space. Rather than relying on simple keyword matching, feature extraction captures deep relational similarities across complex phrasing, synonyms, and multi-sentence structures. These dense representations serve as the computational backbone for vector databases, semantic search engines, nearest-neighbor recommendation algorithms, and clustering pipelines. Text feature extraction is the core retrieval component in almost every modern retrieval-augmented generation architecture.

## Representative Model: `elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF`

### Model profile
This is a 13.15B-parameter logical model (Gemma 4 12B plus LTX projection heads) distributed as GGUF: Q2_K 5.96 GB, Q4_K_M 8.41 GB, Q5_K_M 9.51 GB. Released in August 2026 under the LTX-2.x Community License plus Apache-2.0 for Gemma, it is gated. It is not a general-purpose text embedder; it is the conditioning encoder for LTX-2.5 video generation.

### Architecture and task tailoring
It is an LTX-specific Gemma 4 12B encoder (48 layers) with both LTX audio and video heads. The heads emit 4096 video and 2048 audio channels, concatenated to `[1, 8, 6144]` for the video DiT. All quants match the BF16 source's 686 tensors, shapes, and 13,147,985,090 elements, with ComfyUI state-dict names and an embedded tokenizer. See [LTX family](roadmap.md#model-family-notes).

### Niche mechanism
This is not a pooling embedder: the Gemma 4 12B decoder's final states feed two LTX projection heads that emit 4096 video and 2048 audio conditioning channels, concatenated to `[1,8,6144]` for the DiT's cross-attention. The 48-layer encoder is LTX-specific, not vanilla Gemma, and the quantizer only compressed it. **Design takeaway:** for generative conditioning, keep the text encoder's token sequence and train per-modality projections; do not mean-pool.

### Training and task tailoring
The quantizer did no training; this is a static quantization of `Lightricks/LTX-2.5`'s BF16 `gemma4-12b-with-proj-ltx-2.5` encoder. Metadata identifies the LTX-specific `gemma4-12b-ltx-v1`, not vanilla Gemma 4. Upstream training is not documented.

### How it is used
Use ComfyUI 0.33.0 with a tested ComfyUI-GGUF build, apply `patches/ComfyUI-GGUF-ltx25-gemma4.patch`, place the file in `models/text_encoders/`, and load via `CLIPLoaderGGUF` type `ltxv` with the matching LTX-2.5 transformer and VAEs. It does not run in llama.cpp.

### Fit and caveats
`feature-extraction` is a proxy tag: this is a model-specific conditioning encoder, not a general embedding model. All quants passed a structural audit and produced finite `[1,8,6144]` output; cosine versus BF16 is 0.8698 (Q4_K_M) and 0.8814 (Q2_K). Only Q5_K_M completed end-to-end GPU generation, and Q2/Q4 were not validated end-to-end. It needs a patched ComfyUI and is license-restricted.

## Related Reading
- [Roadmap: adapters, LoRA, and quantization](roadmap.md#adapters-lora-and-quantization) and [LTX family](roadmap.md#model-family-notes).
- [Image-to-Video](image-to-video.md) · [Video-to-Video](video-to-video.md) · [Sentence Similarity](sentence-similarity.md)
