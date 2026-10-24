# Text-to-Speech (`text-to-speech`)

## Overview
Text-to-Speech models synthesize natural, expressive human spoken audio waveforms from written textual inputs. Modern neural architectures employ flow matching, diffusion pipelines, or discrete audio language models coupled with neural vocoders to produce lifelike prosody, emotional nuance, and breathing pauses. The system converts phoneme sequences into continuous acoustic representations with precise duration and pitch control. By conditioning on speaker reference embeddings, modern systems support instantaneous voice cloning and cross-lingual vocal transfer. Text-to-Speech powers accessibility screen readers, automated audiobook narration, virtual assistant voices, video game character dialogue, and real-time voice translation systems.

## Representative Model: `fishaudio/drama-3`

### Model profile
Drama-3 is a 35B-parameter (3B active) sparse MoE text-to-speech model released in October 2026 (arXiv 2610.18010). It ships as BF16 plus FP8 E4M3 experts under the Fish Audio Community License v1.0, which permits commercial use below USD 500k revenue and treats hosted TTS separately; the metadata says "fish-audio-research-license," so the terms conflict. It generates 44.1 kHz speech, supports multi-speaker dialogue, voice cloning, editing, and streaming.

### Architecture tailored to expressive TTS
It is initialized from Qwen3.5: a 40-layer hybrid Gated DeltaNet/full-attention MoE with 2048 hidden and 256 experts top-8. There is no audio encoder; a 13-layer codebook-MTP decoder (4 shared plus 9 per-codebook) predicts 10 codebooks at about 21.5 fps. Predicting multiple codebooks in parallel is what enables low-latency, high-fidelity speech. See [audio codecs](roadmap.md#vision-and-audio-encoders).

### Niche mechanism
Drama-3 has no audio encoder: a 40-layer Qwen3.5 MoE predicts 10 RVQ codebooks at 21.5 fps with a 13-layer codebook-MTP decoder built from 4 shared layers plus 9 codebook-specific components. Generating all codebooks per frame in parallel is what keeps latency low; speaker tags and reference clips condition the same token stream. **Design takeaway:** for TTS, a shared trunk + per-codebook heads is cheaper than ten independent autoregressive models.

### Training and task tailoring
Pretraining spans several hundred billion tokens over three stages (8K→16K context), about 80 languages, and the Muon optimizer. One 32K-token instruction SFT stage follows, then RL with a learned Bradley-Terry preference reward model (no rule-based rewards, KL, or weight decay). The RL stage targets naturalness and instruction following.

### How it is used
Serve with SGLang-Omni or vLLM-Omni and call `/v1/tts` with `text`, `references`, and optional `previous_takes`. Use `<|speaker:0|>`…`<|speaker:15|>` tags and expressive tags such as `[sighs]`. Sampling is 1.0/0.9/30.

### Fit and caveats
The tag fits, but this is a custom `FishOmniModel`, not a standard Transformers pipeline. Reported Seed-TTS CER is 0.83 (zh) and WER 1.54 (en), flagged first takes 1.24%/1.84%, speaker similarity 0.605→0.633 with more references, UTMOS 4.31, and 88.9 ms p50 TTFA at RTF 0.131 on an H200. These are vendor-run, and the download is large.

## Related Reading
- [Roadmap: audio encoders](roadmap.md#vision-and-audio-encoders) and [mixture-of-experts](roadmap.md#mixture-of-experts).
- [Text-to-Audio](text-to-audio.md) · [Audio-to-Audio](audio-to-audio.md) · [Audio-Text-to-Text](audio-text-to-text.md)
