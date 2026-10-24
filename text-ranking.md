# Text Ranking (`text-ranking`)

## Overview
Text Ranking, also known as reranking, evaluates a list of candidate documents against a search query to sort them in order of relevance. Unlike dual-encoder embedding models which compute query and document vectors independently, text ranking models typically use cross-encoders where the query and document are passed jointly through full self-attention layers. This joint attention captures intricate token-to-token interactions, delivering substantially higher precision. Rerankers are standard components in multi-stage search pipelines, placed immediately after vector retrieval to filter and order top results.

## Representative Model: `nvidia/Nemotron-3.5-Rerank-8B-BF16`

### Model profile
Nemotron-3.5-Rerank-8B is an 8.38B BF16 cross-encoder (four shards, ~16.8 GB) released in October 2026 under OpenMDW 1.1, with a Ministral-3 Apache-2.0 base. It takes a text query plus a document that can be text, one image, or image+text and returns a single relevance logit. It supports 34 text languages and 18 image languages.

### Architecture tailored to multimodal reranking
A 24-layer Pixtral vision encoder (1024 hidden, patch 14) feeds a bidirectional Ministral3 tower (34 layers, 4096 hidden, 32 Q/8 KV heads, no causal mask). Mean pooling produces one logit over 32,768 tokens; images are resized to 1,120 px on the long edge (up to 1,540). Removing the causal mask lets query and document tokens see each other, which is the core cross-encoder advantage over bi-encoder embeddings. See [contrastive learning](roadmap.md#contrastive-and-metric-learning).

### Niche mechanism
The cross-encoder removes the causal mask, so query and document tokens attend bidirectionally in one 34-layer tower; mean-pooling the final states and a single linear logit produce a relevance score. Pixtral tokens are injected for page images, and one image per document is allowed. **Design takeaway:** a reranker is a bidirectional encoder + one logit; joint attention is the quality gain over a bi-encoder.

### Training and task tailoring
It initializes from Ministral-3-8B-Instruct-2512-BF16 and trains in two stages: 21M text samples, then 2.3M image and image+text samples. Public data includes ArxivQA, ColPali-PDF, InfoVQA, TAT-DQA, Docmatix, and wiki-ss-nq; synthetic data comes from Nemotron, Qwen, Gemma, Pixtral, and gpt-oss. This multimodal mix is the task-specific adaptation.

### How it is used
Use Transformers 5.20 + FlashAttention 2 with `AutoModelForSequenceClassification` and `query`/`document` roles, sentence-transformers `CrossEncoder` 6.1, or vLLM 0.32's `/classify` with `use_activation=False`. BF16 is standard, and one image is allowed per document.

### Fit and caveats
The tag undersells the multimodal input; the output is a raw logit, not a probability. Reported ViDoRe v3.1 NDCG@10 is 67.22/66.89/67.56 for image/text/image+text versus Qwen3-VL-Reranker-8B at 66.02/64.88/66.03; MIRACL-VISION is 78.69 vs 75.06, claimed SOTA as of 20 October 2026. Use `/classify`, not `/score` or `/rerank`, with images; FlashAttention 2 on Transformers 5.17–5.19 has image-path bugs.

## Related Reading
- [Roadmap: contrastive learning](roadmap.md#contrastive-and-metric-learning) and [Mistral family](roadmap.md#model-family-notes).
- [Visual Document Retrieval](visual-document-retrieval.md) · [Sentence Similarity](sentence-similarity.md) · [Feature Extraction](feature-extraction.md)
