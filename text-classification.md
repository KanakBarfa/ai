# Text Classification (`text-classification`)

## Overview
Text Classification is an essential natural language processing task that categorizes complete textual passages, documents, or queries into predefined classes. The model processes input tokens through bidirectional encoders or autoregressive feature representations, condensing the semantic meaning into a final classification vector. Common implementations include sentiment analysis, spam filtering, legal document classification, customer ticket routing, language identification, and toxic comment moderation. Text classification is one of the most widely deployed machine learning tasks in production, powering enterprise search relevance, helpdesk triage, and social media feed monitoring.

## Representative Model: `baselabs/mammoth-1`

### Model profile
Mammoth-1 is a 27.36B BF16 model (about 109 GB) released in October 2026 under MIT. It takes text plus optional images and questions/options and returns calibrated probabilities for every option from one shared-prefix prefill. The card is English-only. It is a decision model with a custom engine, not a standard sequence-classification head.

### Architecture tailored to multi-option decisions
It is a Qwen3.5-27B derivative: 64 layers, 5120 hidden, 24 Q/4 KV heads, a 3:1 linear/full attention pattern, and partial RoPE, plus a frozen 27-layer/1152-wide vision encoder. Full attention is made block-bidirectional, and a custom `slot_lm` readout with 256 slots scores all options in one pass. That shared-prefix design is what makes per-option probabilities cheap. See [Qwen family](roadmap.md#model-family-notes).

### Niche mechanism
Mammoth keeps the Qwen3.5-27B backbone but makes full-attention blocks bidirectional and replaces the LM head with a 256-slot `slot_lm` readout. All options share one prefix prefill; each slot reads the final hidden state and produces a calibrated score, so scoring 100 options costs one forward pass, not 100. CE + Brier loss trains the probabilities and an explicit "I don't know" slot. **Design takeaway:** for multi-option decisions, score in one prefill with a dedicated readout and calibrate; do not generate each answer.

### Training and task tailoring
The language model is fully fine-tuned while the vision tower stays frozen. Training uses 6.21M examples (58.3% core decision, 16.1% safety/abstention, 16.1% judging/expert, 9.5% public) with cross-entropy plus Brier loss (λ=1) and a trained "I don't know" option. It ran 9,859 steps at LR 2.7e-6 with AdamW/BF16 on 72 GB300s.

### How it is used
Use the Baseten API or OneJev `DecisionEngine`/`SystemOneRequest` with `noul`/`choice`/`score` question types. Maximum input is 16,384 tokens; overlong requests are refused rather than truncated. BF16 is standard.

### Fit and caveats
The `text-classification` tag is loose: this is a multimodal decision model with a custom engine/readout, not a standard HF classifier. It is 109 GB, English-only, and its benchmark is vendor-run. Reported Jev Decision Index 0.4 is 71.24, #1 of 208, ahead of Perplexity Decider v1.2 (69.42) and Jev 1.14 (62.2); the Artificial Analysis replica is 65.9 vs 65 for d1-ultra, leading 4 of 5 Jev areas.

## Related Reading
- [Roadmap: Qwen family](roadmap.md#model-family-notes) and [post-training](roadmap.md#post-training-sft-preferences-and-rl).
- [Zero-Shot Classification](zero-shot-classification.md) · [Text Ranking](text-ranking.md) · [Token Classification](token-classification.md)
