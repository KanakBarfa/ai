# Visual Document Retrieval (`visual-document-retrieval`)

## Overview
Visual Document Retrieval is the task of searching, indexing, and ranking visually structured documents based on textual or image-based queries. Traditional retrieval systems rely strictly on plain extracted text, which strips away structural cues like headers, tables, charts, and brand marks. Visual Document Retrieval systems encode the complete rendered page image, capturing spatial formatting, visual typography, and graphical elements directly into dense embedding representations. This enables retrieval engines to surface relevant slide decks, schematics, and annual reports that pure keyword search fails to match. Core applications include enterprise knowledge discovery, patent exploration, academic paper indexing, and multimodal retrieval augmented generation.

## Representative Model: `nvidia/Nemotron-3.5-Embed-8B-BF16`

### Model profile
Nemotron-3.5-Embed-8B is an 8.38B BF16 embedding model released in October 2026 under the OpenMDW-1.1 license. It turns a text query plus a document (text, page image, or image+text) into one 4096-dimensional vector. It supports 34 text languages and 18 visual-document languages, with Matryoshka truncation to 1024 or 512 dimensions.

### Architecture tailored to visual documents
It uses the Ministral-3-8B backbone (34 layers, 4096 hidden, 32 Q/8 KV heads, 262K positions) plus a 24-layer Pixtral-style vision encoder. Image tokens are merged spatially, then mean-pooled and L2-normalized into the embedding. The query side is text-only; the document side carries both layout pixels and extracted text. The design is a bi-encoder, so pages can be embedded offline and searched with cosine similarity. Background concepts are in [contrastive learning](roadmap.md#contrastive-and-metric-learning).

### Niche mechanism
Nemotron embeds a document as a page image + text through Ministral3 + Pixtral, mean-pools the final states, and L2-normalizes to 4096-d; The embedding supports Matryoshka truncation to 1024/512-d. The text LM is frozen and only the vision/projector path is trained, so text embeddings stay aligned with the parent. **Design takeaway:** freeze the text embedder, adapt vision, and use MRL for storage; queries stay text-only.

### Training and task tailoring
The language model starts from the text-only Nemotron-3-Embed-8B and is frozen. The vision encoder and projector are trained on 2.3M image samples plus 50M text samples, using synthetic LLM-generated queries over public documents. Because only the multimodal path is adapted, text embeddings stay aligned with the parent model.

### How it is used
Use `SentenceTransformer` with `encode_query`/`encode_document`, or Transformers with `AutoModel`/`AutoProcessor` and `query:`/`passage:` prefixes. vLLM pooling is supported. BF16 with FlashAttention 2, 32K token limit, and images resized to about 1,120 px on the long edge.

### Fit and caveats
The tag fits, but the model returns embeddings rather than answers, queries cannot be images, and 32K/BF16 needs a serious GPU. Reported ViDoRe V3 NDCG@10 is 60.39 (image), 60.65 (text), 61.47 (image+text), and MIRACL-VISION is 72.64; these are vendor numbers.

## Related Reading
- [Roadmap: contrastive learning](roadmap.md#contrastive-and-metric-learning) and [Mistral/Ministral family](roadmap.md#model-family-notes).
- [Text Ranking](text-ranking.md) · [Sentence Similarity](sentence-similarity.md) · [Document Question Answering](document-question-answering.md)
