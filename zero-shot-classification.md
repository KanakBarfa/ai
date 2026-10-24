# Zero-Shot Classification (`zero-shot-classification`)

## Overview
Zero-Shot Classification enables the categorization of text into custom target labels without requiring any labeled training data for those specific classes. The task is typically formulated as natural language inference, where the input text is treated as a premise and candidate labels are formatted into hypothesis statements such as this text is about sports. The model calculates the probability of entailment for each hypothesis. This approach allows developers to create instant classifiers on the fly for dynamic topic labeling, user intent detection, and urgent content filtering without data collection delays.

## Representative Model: `vllm-sr/d3-turbo`

### Model profile
d3-turbo is a 26.09B BF16 decision model released in October 2026 under Apache-2.0, including a 0.46B vision tower and a 1.3M FP32 readout. It accepts text/JSON with optional images or videos plus typed questions and returns probabilities per option, boxes, or tracks. It does not generate free text.

### Architecture tailored to option scoring
It uses a Qwen3.5-family 64-layer backbone with a 3:1 linear/full attention pattern, 5120 hidden, 24 Q/4 KV heads, head_dim 256, and a 27-layer/1152-wide vision tower. A 255-code readout replaces the LM head, with `prefix_bidir`/StateOnce and `choice`, `noul`, `score`, `box`, and `track` question types. Scoring all candidate codes in one forward pass is what makes it fast. See [Qwen family](roadmap.md#model-family-notes).

### Niche mechanism
d3-turbo replaces the LM head with a 255-code readout: the model reads the shared prefix and emits a code for each typed question, so 255 questions cost one 2.12 s forward pass instead of 255 generations. `prefix_bidir`/StateOnce makes the option-scoring prefix bidirectional and cache-efficient. **Design takeaway:** if the answer space is finite, score codes in one prefill; do not generate text.

### Training and task tailoring
The card describes it as an upgraded 27B Decision 3.0 model built on Qwen3.8-27B with an answer-code readout. Data, objectives, and SFT/RL stages are not stated, and there is no technical report. This is therefore a fixed-code decision model, not NLI-style zero-shot classification.

### How it is used
Use `AutoModel.from_pretrained(..., trust_remote_code=True)` plus `model.system_one(...)`, or the vLLM SR Engine `/v1/systemone`. BF16 is standard; the default budget is 32,768 tokens per question and 131,072 with `long_context=True`. It supports up to 255 questions, images up to 1.6 MP, and video at 2 FPS/32 frames.

### Fit and caveats
The tag is loose: this is a trained decision model, not an entailment-based zero-shot classifier. It needs custom code and recent/nightly vLLM, emits no text, and caps questions at 255. Reported text Jev index is 70.2 (own estimate) vs 66.3 for d3, vision 72.9 vs 71.3, box 0.90 at IoU 0.5 with AUROC 0.96, tracking AUC 0.79/0.80, presence F1 0.97, and 255 questions in 2.12 s vs 32.94 s for d3-flash.

## Related Reading
- [Roadmap: Qwen family](roadmap.md#model-family-notes) and [task-tag caveats](roadmap.md#task-tag-caveats).
- [Text Classification](text-classification.md) · [Zero-Shot Image Classification](zero-shot-image-classification.md) · [Text Ranking](text-ranking.md)
