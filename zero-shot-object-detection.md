# Zero-Shot Object Detection (`zero-shot-object-detection`)

## Overview
Zero-Shot Object Detection locates and classifies objects within visual scenes using arbitrary natural language descriptions provided at test time, without prior supervised training on those specific object categories. The architecture fuses visual feature representations with linguistic tokens, enabling the detector to ground free-form text queries into precise bounding box coordinates. This open-vocabulary capability allows users to search images for obscure, highly specialized, or rare items simply by typing descriptive phrases. Typical use cases include open-world robotic manipulation, dynamic visual search engines, smart home security monitoring, and automated forensic review of large image archives.

## Representative Model: `omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps`

### Model profile
This is a 3.755B BF16 checkpoint (about 7.5 GB) built by applying VLM-R1 reinforcement learning to Qwen2.5-VL-3B-Instruct for referring expression comprehension. Uploaded in early 2025 and updated in April 2025, it declares no license. It outputs a single box as generated JSON, not a detection tensor.

### Architecture tailored to text-grounded boxes
The base Qwen2.5-VL-3B has a 36-layer, 2048-hidden language model (16 Q/2 KV heads, MRoPE, 128K context) and a 32-layer ViT (1280 hidden, patch 14, 2×2 merge). The task-specific behavior comes from RL, not a new head: the model generates `<think>...</think><answer>{"box": [...]}</answer>`, and the reward combines box IoU with format compliance. See [multimodal fusion](roadmap.md#multimodal-fusion).

### Niche mechanism
VLM-R1 does not add a detection head: it uses GRPO to teach Qwen2.5-VL to emit `{"box":[x1,y1,x2,y2]}` in an `<answer>` block, with reward = IoU + format. The model learns grounding as a text-generation policy over the 32-layer ViT + 36-layer LLM stack. **Design takeaway:** RL with an IoU verifier can turn a VLM into a single-box detector; it is not open-vocabulary multi-object detection.

### Training and task tailoring
GRPO (VLM-R1) starts from the Instruct model and trains on RefCOCO/+/g for two epochs; this checkpoint is step 500. Hyperparameters include N=8 rollouts, temperature 0.9, KL 0.04, and learning rate 1e-6. The Hub metadata lists RefCOCOg only, which conflicts with the paper's broader training mix.

### How it is used
Load `Qwen2_5_VLForConditionalGeneration` with its processor and prompt: "Please provide the bounding box coordinates of the region this sentence describes: {query}." Use BF16 with FlashAttention 2 and near-greedy decoding (temperature 0.1, top_k 1, top_p 0.001, repetition penalty 1.05).

### Fit and caveats
The tag implies open-vocabulary detection, but this is single-box referring expression comprehension. It is text-output, trained for only 500 RL steps, and has no license. Reported results are RefCOCO val 90.0, RefCOCO+ 83.6, RefCOCOg 87.15, and LISA-Grounding 61.88, versus SFT 88.85/82.3/85.95/55.66.

## Related Reading
- [Roadmap: multimodal fusion](roadmap.md#multimodal-fusion) and [post-training](roadmap.md#post-training-sft-preferences-and-rl).
- [Object Detection](object-detection.md) · [Image-Text-to-Text](image-text-to-text.md) · [Visual Question Answering](visual-question-answering.md)
