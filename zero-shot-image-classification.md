# Zero-Shot Image Classification (`zero-shot-image-classification`)

## Overview
Zero-Shot Image Classification categorizes images into novel classes without requiring any task-specific labeled training data for those categories. The model operates by aligning visual embeddings from an image encoder with text embeddings from a language encoder within a shared multimodal latent space. During inference, arbitrary candidate class names are formatted into natural language prompts, and the model selects the label whose textual embedding exhibits the highest cosine similarity to the image embedding. This flexibility eliminates the need to collect thousands of domain-specific images and retrain classifiers, dramatically accelerating deployment for visual search, dynamic inventory categorization, and open-vocabulary classification.

## Representative Model: `knowledgator/gliclass-v4-27b`

### Model profile
GLiClass v4 27B is a 26.997B BF16 decision model (about 54 GB) released in October 2026 under Apache-2.0. It accepts text or an image plus candidate labels or an instruction and returns calibrated label scores. It supports single-label and multi-label decisions, up to about 25 labels, and is English-centric.

### Architecture tailored to label decisions
It is built on a Qwen3.8-27B decoder (64 layers, 5120 hidden, 24 Q/4 KV, full attention every fourth layer). Candidate labels become transient `<<LABEL>>` tokens that read the decoder's KV cache, and a two-layer scorer plus MLP scores each pair; a per-pair temperature/bias calibrator is applied afterward. Recurrent reasoning can run up to three steps. This avoids running a separate forward pass per label, which is why it can score many labels quickly. See [transformers](roadmap.md#transformers-and-attention).

### Niche mechanism
GLiClass injects labels as transient `<<LABEL>>` tokens that read the decoder's KV cache, then a 2-layer scorer + MLP produces one logit per (input, label). A per-pair temperature/bias calibrator maps logits to probabilities; recurrence can refine up to three steps. **Design takeaway:** label tokens + KV-cache readout scale to many labels without a forward pass per label; calibrate per pair.

### Training and task tailoring
Training has three stages: a supervised label-decision mixture covering classification, NLI, QA, reranking, tools, moderation, judging, and reasoning (focal loss); PPO-style RL on task metrics such as F1/accuracy/recall; then NLL calibration with the backbone frozen. This is not CLIP-style image-text contrastive zero-shot classification.

### How it is used
Install `gliclass`, then use `GLiClassModel` with `ZeroShotClassificationPipeline`. Run in BF16 with a prompt and optional image; context runs from 8K to 32K tokens. Multi-label and single-label scoring modes are both supported.

### Fit and caveats
The tag is only approximate: this is a general decision model, not a vision-language embedding matcher. The card is partly copied from the 9B model ("9B" text, 55.6 skill, and 34 ms belong to the 9B), while 27B weights score 61.74 at 73.7 ms. The scorer is a bge-small BERT per config despite DeBERTa wording, and English dominance limits multilingual use.

## Related Reading
- [Roadmap: transformers](roadmap.md#transformers-and-attention) and [post-training](roadmap.md#post-training-sft-preferences-and-rl).
- [Text Classification](text-classification.md) · [Zero-Shot Classification](zero-shot-classification.md) · [Image Classification](image-classification.md)
