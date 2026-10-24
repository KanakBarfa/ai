# Robotics (`robotics`)

## Overview
Robotics models translate multimodal sensory perceptions such as camera feeds and depth maps into physical control actions and actuator trajectory signals. Recent state-of-the-art robotics systems employ Vision-Language-Action models that connect multimodal foundational reasoning directly to robot arm joint commands and end-effector coordinates. This allows robots to follow natural language commands to pick, manipulate, and assemble objects in unstructured real-world settings. The models generalize physical mechanics across diverse objects without manual reprogramming. Key applications include warehouse automation, automated surgical robotics, assistive domestic care robots, and industrial assembly lines.

## Representative Model: `sensenova/sensenova-roborsi-27b-preview`

### Model profile
SenseNova-RoboRSI-27B-Preview is a 27.36B BF16 vision-language model (12 shards, ~54.8 GB) released in October 2026 under Apache-2.0. It takes robot camera images and text and returns text plans, action decisions, feedback, and tool calls. It is a System-2 planner, not a vision-language-action controller.

### Architecture tailored to robot planning
It is built on Qwen3.5-27B: 64 layers with a 3:1 linear/full attention pattern, 5120 hidden, 24 Q/4 KV heads, head_dim 256, a 27-layer/1152-wide vision tower, and 262K positions. `reasoning_effort` can be low/medium/xhigh, with medium recommended. The language-side planning is the point: the model reasons over camera views and emits high-level actions, while a separate VLA or SAM3 stack executes them. See [Qwen family](roadmap.md#model-family-notes).

### Niche mechanism
RoboRSI is a System-2 planner: it sees camera frames and text, reasons with the 3:1 Gated DeltaNet/full-attention Qwen3.5-27B stack, and emits plans/tool calls for a separate VLA controller. The post-training loop validates traces in simulation and on real robots and feeds corrections back into the model. **Design takeaway:** split slow planning from fast control; the planner needs visual feedback and a harness, not joint torques.

### Training and task tailoring
It is post-trained from Qwen3.8-27B via a joint model-harness RSI loop. Validated interaction traces from simulation and real robots supervise goal understanding, action decisions, and feedback correction; the model returns to the harness for evaluation. Data volumes and the SFT/RL schedule are not stated, and the technical report is pending.

### How it is used
Serve with vLLM 0.21.0 in BF16, TP=4, `--max-model-len 262144`, `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`, and `--mamba-cache-mode align`; call chat completions with `reasoning_effort="medium"`. The RoboRSI Harness is required for evaluation, with VLA/SAM3 optional.

### Fit and caveats
The tag fits, but this is a planner that emits text/tool calls, not a controller. It needs the harness and usually a VLA/SAM3 stack. RoboDojo evaluation is not yet supported, and the report is pending. Reported RoboDojo average is 26.13 with 20.83% success under VLA-assisted execution (2.84 points below GPT-6 Astra); LIBERO-Pro reaches 92.5% VLA-assisted and 95.0% direct VLM, and RoboCasa365 improves from 26.0% to 46.5% while cutting tokens from 59.5K to 15.9K.

## Related Reading
- [Roadmap: Qwen family](roadmap.md#model-family-notes) and [post-training](roadmap.md#post-training-sft-preferences-and-rl).
- [Reinforcement Learning](reinforcement-learning.md) · [Video-Text-to-Text](video-text-to-text.md) · [Keypoint Detection](keypoint-detection.md)
