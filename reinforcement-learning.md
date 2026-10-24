# Reinforcement Learning (`reinforcement-learning`)

## Overview
Reinforcement Learning models train autonomous agents to make sequences of optimal decisions by interacting with dynamic environments to maximize cumulative reward signals. Modern approaches include policy gradient networks, deep Q-networks, actor-critic frameworks, and decision transformers that formulate reinforcement learning as autoregressive trajectory modeling. Rather than learning purely from static labeled datasets, RL agents learn through exploration, feedback, and policy refinement. Key applications include algorithmic trading, chip design floorplanning, industrial cooling optimization, autonomous navigation, and post-training alignment for large language models.

## Representative Model: `self-hpo/Self-HPO-Qwen3.5-35B-A3B`

### Model profile
Self-HPO is a 35.95B-parameter (35B total, 3B active) BF16 coding-agent policy (about 71.9 GB) released in October 2026 under Apache-2.0. It is a text-to-text model; the config carries a vision tower that is unused for its software-engineering work. The `reinforcement-learning` tag records how it was trained, not a standard Gym-style environment.

### Architecture and training
It is a Qwen3.5 MoE: 40 layers, 2048 hidden, 256 routed experts with top-8 plus a 512-wide shared expert, a 3:1 Gated DeltaNet/full-attention pattern, 16 Q/2 KV heads, head_dim 256, and 262K context. Training starts from Qwen3.5-35B-A3B and alternates an H-step, where the policy proposes six harness revisions from its own failure traces, with a P-step of GRPO plus Balanced Adaptive Rollout using binary executable-verifier rewards. It runs two rounds and 50 RL updates per P-step on 2,484 tasks; unsolved tasks seed the next H-step. No SFT stage is reported. See [post-training](roadmap.md#post-training-sft-preferences-and-rl).

### Niche mechanism
The "policy" includes the harness: an H-step asks the model to propose six revisions to its own agent scaffold from failure traces, then a P-step runs GRPO with binary executable-verifier rewards and balanced adaptive rollout. Two rounds, 50 updates per P-step, 2,484 tasks; unsolved tasks seed the next H-step. **Design takeaway:** when the environment is part of the solution, optimize model and harness together; verifiable rewards make GRPO practical.

### How it is used
Serve with `scripts/eval/serve_model.sh` (SGLang, 8 GPUs) and run `scripts/eval/gen_traj.sh --harness self-hpo`. It requires the Self-HPO/Orchard harness and sandbox; without them the checkpoint underperforms.

### Fit and caveats
The tag is about training provenance, not a reusable RL environment. Results are self-reported avg@3 on SWE-bench Verified/Multi/Pro: 61.0/54.2/58.3 (57.8 avg) for the base, 69.5/59.7/65.2 (64.8) for Self-HPO, 66.3 for an external-proposer HPO variant, and 61.9 avg for RLVR. It is SWE-only, harness-coupled, and has no quantized release.

## Related Reading
- [Roadmap: post-training](roadmap.md#post-training-sft-preferences-and-rl) and [mixture-of-experts](roadmap.md#mixture-of-experts).
- [Text Generation](text-generation.md) · [Robotics](robotics.md) · [Video-Text-to-Text](video-text-to-text.md)
