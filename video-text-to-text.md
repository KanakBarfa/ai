# Video-Text-to-Text (`video-text-to-text`)

## Overview
Video-Text-to-Text models process entire sequential video streams alongside textual queries to generate reasoned, descriptive textual responses. Processing video requires modeling both individual frame semantics and continuous temporal relationships across time, allowing the system to comprehend actions, detect scene transitions, track evolving narratives, and identify fleeting visual events. These architectures sample video frames, encode spatiotemporal features, and feed them into language decoders to perform video summarization, timestamps localization, action explanation, and video question answering. Key deployment areas include automated sports commentary, security surveillance analytics, media cataloging, and autonomous driving scenario reasoning.

## Representative Model: `nvidia/Flora-Rewarder`

### Model profile
Flora-Rewarder is a 27.8B video-text reward model released in October 2026 under the NVIDIA OneWay Noncommercial License. It takes multiview robot or human egocentric video plus a text prompt and returns textual scores. It is not a conversational video model; it is a reward model for evaluating robot trajectories.

### Architecture tailored to reward scoring
It is built on a Qwen3.5-27B VLM: 64 text layers with a 3:1 linear-attention/full-attention pattern, 5120 hidden, 24 Q/4 KV heads, and a 27-layer/1152-hidden vision tower. The prompt selects one of three modes—outcome reward, dense reward, or rubric reward—and the decoder emits the score as text. This prompt-conditioned scoring is the task-specific adaptation; there is no separate regression head. See [Qwen family](roadmap.md#model-family-notes).

### Niche mechanism
Flora is a VLM used as a reward model: the prompt selects outcome/dense/rubric mode, and the Qwen3.5-27B decoder writes a score as text rather than using a regression head. Video frames are sampled at 1–10 fps depending on mode, and the model is SFT'd on mixed human/automatic/VLM labels. **Design takeaway:** a prompted VLM can replace a bespoke reward head; the trade-off is parsing and calibration.

### Training and task tailoring
Training is SFT only, with output-token cross-entropy for one epoch. The language model, visual merger, and vision encoder are all fine-tuned (the encoder at a lower learning rate). Outcome, dense, and rubric rows are shuffled and trained jointly. Labels combine human, automatic, and VLM annotation, totaling fewer than 1M examples and under 10,000 hours of English video.

### How it is used
Run with Transformers in BF16 on an H100. Outcome mode accepts up to 64 frames per episode (about 10 frames/s); dense mode uses 16 s, 64-frame windows at 4 fps; rubric mode samples each camera at 1 fps (≤82 frames). Token budgets are 16,384 or 32,768, and the textual output must be parsed.

### Fit and caveats
`video-text-to-text` is misleading: this emits scores, not descriptions or answers. It is non-commercial, English-only, SFT-only, and has no separate test set. Reported numbers are RoboRewardBench MAE 40.0 (×100 on a 1–5 scale) and RobometerBench OOD Kendall's tau-a 72.0, with a claim of ~125× lower cost than GPT-6 Astra.

## Related Reading
- [Roadmap: post-training](roadmap.md#post-training-sft-preferences-and-rl) and [task-tag caveats](roadmap.md#task-tag-caveats).
- [Robotics](robotics.md) · [Video Classification](video-classification.md) · [Reinforcement Learning](reinforcement-learning.md)
