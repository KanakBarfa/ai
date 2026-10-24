# Any-to-Any (`any-to-any`)

## Overview
Any-to-Any models represent the frontier of unified multimodal artificial intelligence. These architectures accept arbitrary combinations of sensory inputs including text, audio, images, and video, and can synthesize outputs across any requested target modality. Instead of chaining separate specialized models together, an Any-to-Any network processes different data streams within a shared representation space or unified discrete tokenization framework. This enables rich cross-modal reasoning, such as listening to spoken instructions while inspecting an image and responding with both synthetic speech and a generated diagram. Common use cases include next-generation conversational agents, real-time multimedia creative assistants, and cross-modal translation systems.

## Representative Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`

### Model profile
Qwen3-Omni is a natively omni-modal model released in September 2025. The checkpoint stores about 35.26B BF16 parameters (the "30B-A3B" name refers to activated compute), accepts interleaved text, images, audio, and video, and returns text or speech. It covers 119 text languages, 19 speech-input languages, and 10 speech-output languages. The card carries conflicting license fields (`other` with `license_name: apache-2.0`), so verify terms before commercial use.

### Architecture tailored to any-to-any
It uses a Thinker–Talker MoE design. The Thinker is a 48-layer MoE transformer (128 experts, top-8) that fuses a 27-layer ViT (deepstack features at layers 8/16/24) and a from-scratch 32-layer AuT audio encoder. The Talker is a separate 20-layer MoE (128 experts, top-6) feeding a multi-codebook speech decoder built on a 16-quantizer Code2Wav ConvNet. This split lets reasoning stay text-centric while speech generation runs in parallel for low latency. Shared concepts are in the [roadmap](roadmap.md#multimodal-fusion).

### Niche mechanism
Thinker–Talker split: the Thinker (48L, 128 experts top-8) fuses ViT/AuT tokens and emits text; the Talker (20L, top-6) reads the Thinker's last hidden state plus text and produces 16 codebooks of speech tokens, decoded by Code2Wav. Deepstack injects ViT layers 8/16/24 into the LLM at several depths, and AuT is trained from scratch at 12.5 Hz. The parallel Talker path gives streaming speech without blocking text reasoning. **Design takeaway:** decouple semantics from waveform generation, and share context by conditioning, not by sharing every weight.

### Training and task tailoring
Pretraining has three stages: first freeze the LLM and train encoders/adapters, then unfreeze on roughly 2T tokens (text 0.57T, audio 0.77T, image 0.82T, video 0.1T), then extend context to 32K. AuT saw 20M hours of supervised audio. Post-training uses three-stage Thinker SFT plus GSPO with rule/model rewards, and a four-stage Talker recipe (speech SFT, continued pretraining, DPO, speaker fine-tuning).

### How it is used
Serve the Thinker with vLLM (audio output was pending at card time) or run Transformers with `qwen-omni-utils` and ffmpeg. Weights are BF16 with FlashAttention 2; `disable_talker()` saves about 10 GB. Media is interleaved in ChatML-style messages, and a system prompt selects voice and behavior.

### Fit and caveats
The `any-to-any` label overstates the output: this model produces text and speech, not images or video. It is excellent for audio-video understanding and voice conversation, but the 35B stored size, custom serving path, and license ambiguity matter for deployment.

## Related Reading
- [Roadmap: multimodal fusion](roadmap.md#multimodal-fusion) and [mixture-of-experts](roadmap.md#mixture-of-experts).
- [Audio-Text-to-Text](audio-text-to-text.md) · [Image-Text-to-Text](image-text-to-text.md) · [Text-to-Speech](text-to-speech.md)
