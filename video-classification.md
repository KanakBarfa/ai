# Video Classification (`video-classification`)

## Overview
Video Classification assigns high-level categorical labels to continuous video clips by jointly analyzing spatial visual features and temporal motion dynamics across multiple sequential frames. Models employ divided space-time attention mechanisms, 3D convolutional networks, or temporal transformers to track human actions, camera transitions, and environmental changes. This task goes beyond static image recognition by differentiating actions that look identical in single frames but diverge across time. Practical applications include automated content moderation on video platforms, sports analytics, surveillance anomaly detection, gesture control, and video recommendation indexing.

## Representative Model: `nvidia/Cosmos-Embed1-448p-anomaly-detection`

### Model profile
Cosmos-Embed1-448p anomaly detection is a 1.198B-parameter video-text embedding model (F32, about 18.4 GB) released under the NVIDIA Open Model License and updated in October 2026. It maps video frames or text to a 768-dimensional L2-normalized embedding, so classification is done by cosine similarity rather than a fixed classifier head.

### Architecture tailored to video embeddings
It is a QFormer-based video-text embedder with an EVA-ViT-g vision backbone. Eight frames at 448×448 are concatenated temporally and summarized by 32 visual query tokens through QFormer cross-attention; the pooled output projects to 768 dimensions, and text is encoded to 128 tokens. Temporal concatenation plus learned queries lets one embedding represent an event rather than a single frame. See [contrastive learning](roadmap.md#contrastive-and-metric-learning).

### Niche mechanism
Cosmos-Embed1 samples 8 frames, concatenates them temporally, and lets 32 learned QFormer queries cross-attend to EVA-ViT-g features, compressing a clip into a fixed 768-d vector. Text is encoded separately, and contrastive + captioning losses align the two spaces; classification is cosine ranking, not a softmax head. **Design takeaway:** a QFormer is a fixed-length video tokenizer; contrastive embedding + prompt text gives zero-shot classification without retraining.

### Training and task tailoring
It is fine-tuned from Cosmos-Embed1-448p with NVIDIA TAO (`cosmos_embed1`). LoRA adapts the vision backbone, QFormer, and projection using contrastive and captioning losses. Training uses under 10,000 hours from PhysicalAI-Event-Videos v1.1 at 8 frames and 512×512, with evaluation at 448×448; this domain adaptation targets anomaly/event retrieval.

### How it is used
Load with `AutoModel.from_pretrained(..., trust_remote_code=True)` and `AutoProcessor`. Video input is `[B,3,8,448,448]` and text is `[B,128]`; use `get_video_embeddings(...).visual_proj` and `get_text_embeddings(...).text_proj`. BF16 on an A100 is supported.

### Fit and caveats
The tag says video-classification, but this is an embedding/retrieval model with no classifier logits. Event retrieval scores are 63.21% mAP, 58.88% Hit@1, and 86.97% Hit@5 versus 49.78/44.31/74.74 for the base. Kinetics-400 Top-1 is 83.01% versus 88.23% for the base, so domain tuning costs general action accuracy. Checkpoint selection reused the evaluation gallery, so the domain numbers are not independent.

## Related Reading
- [Roadmap: contrastive learning](roadmap.md#contrastive-and-metric-learning) and [vision encoders](roadmap.md#vision-and-audio-encoders).
- [Video-Text-to-Text](video-text-to-text.md) · [Zero-Shot Image Classification](zero-shot-image-classification.md) · [Text Ranking](text-ranking.md)
