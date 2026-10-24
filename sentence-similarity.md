# Sentence Similarity (`sentence-similarity`)

## Overview
Sentence Similarity models evaluate pairs of text inputs and calculate a numerical score indicating how closely they align in meaning and intent. Typically built on Siamese or dual-encoder transformer frameworks, the model generates embeddings for each sentence and computes their cosine similarity or distance metric. Unlike keyword matching, sentence similarity captures paraphrase relationships, colloquial phrasing, and conceptual overlaps across varied sentence structures. This capability is essential for FAQ matching bots, automated duplicate question detection, plagiarism identification, semantic deduplication, and semantic caching layers across high-throughput production environments.

## Representative Model: `Qwen/Qwen3-VL-Embedding-2B`

### Model profile
Qwen3-VL-Embedding-2B is a 2.13B BF16 multimodal embedder (~4.3 GB) released in January 2026 under Apache-2.0. It maps text, images, screenshots, videos, and mixtures to one embedding vector of up to 2048 dimensions, covering 30+ languages. It is instruction-aware and not a chat VLM.

### Architecture tailored to multimodal similarity
It is built on Qwen3-VL-2B-Instruct: 28 text layers, 2048 hidden, 16 Q/8 KV heads, head_dim 128, FFN 6144, vocab 151,936, plus a 24-layer vision tower (1024 hidden, patch 16, merge 2, deepstack at layers 5/11/17). A pooled Matryoshka projection supports 64–2048 dimensions. This lets one model compare text-text, text-image, and video-text pairs without separate encoders. See [multimodal fusion](roadmap.md#multimodal-fusion).

### Niche mechanism
A single 2B Qwen3-VL encoder handles text, images, and video; deepstack features from ViT layers 5/11/17 go into the LLM, and a pooled projection with Matryoshka training yields one vector truncatable to 64–2048 dims. Distilling the Qwen3-VL reranker into this bi-encoder transfers fine-grained relevance while keeping retrieval fast. **Design takeaway:** bi-encoder for speed, cross-encoder as teacher; MRL lets one index serve multiple dimension budgets.

### Training and task tailoring
Training is multi-stage: large-scale multimodal contrastive pretraining, then distillation from the Qwen3-VL-Reranker cross-encoder, plus instruction tuning and Matryoshka representation learning. The reranker distillation transfers fine-grained relevance judgments into a fast bi-encoder. Data mixtures, batch sizes, and steps are not stated.

### How it is used
Use `SentenceTransformer(...).encode(...)` for text and text+image; the default instruction is "Represent the user's input." Mixed or video input needs the repository's `Qwen3VLEmbedder` or vLLM/SGLang embedding modes. Use BF16 and cosine similarity; it does not generate text.

### Fit and caveats
The tag suggests a text bi-encoder, but this is multimodal, and video/mixed input needs custom code. Context is documented as 32K, Matryoshka truncation trades accuracy, and it is not a chat model. Reported MMEB-V2 All is 73.2 for 2B (Image 75.0, Video 61.9, VisDoc 79.2), the 8B reaches 77.8, and MMTEB mean task is 63.87; instructions add about 1–5%.

## Related Reading
- [Roadmap: contrastive learning](roadmap.md#contrastive-and-metric-learning) and [multimodal fusion](roadmap.md#multimodal-fusion).
- [Feature Extraction](feature-extraction.md) · [Text Ranking](text-ranking.md) · [Visual Document Retrieval](visual-document-retrieval.md)
