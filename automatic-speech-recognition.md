# Automatic Speech Recognition (`automatic-speech-recognition`)

## Overview
Automatic Speech Recognition, or ASR, transcribes human spoken audio into written natural language text. Modern end-to-end architectures utilize conformer networks, hybrid connectionist temporal classification, or encoder-decoder transformers trained on tens of thousands of multilingual speech hours. The system models acoustic phonemes and language context concurrently to handle background noise, accents, and rapid speech. By learning rich linguistic priors, the system robustly resolves homophones and ambiguous phrasing. ASR is the fundamental front-end for voice-controlled assistants, automated meeting transcription, video subtitle generation, call center analytics, and hands-free dictation software.

## Representative Model: `inclusionAI/Ming-3.0-asr-tiny`

### Model profile
Ming-3.0-ASR-tiny is an 8.87B-parameter BF16 speech recognizer (about 36.8 GB) released in October 2026 under MIT, with bundled Apache-2.0 files retaining their notices. It accepts audio, optionally with a text prompt or hotwords, and returns text. It covers Chinese and English plus 15 regional dialect test sets. "Tiny" refers to the family position, not the size.

### Architecture tailored to contextual ASR
It pairs a Whisper-style audio encoder (128 mel bins, 32 layers, 1280 width, 20 heads, 3,000 frames, stride-2 downsampler) with a Ling-tiny-v3 decoder (24 layers, 1536 hidden, 16 heads, 128 experts top-8 plus shared, MLA with q_lora 256/kv_lora 512, 8,192 context, one MTP layer). The prompt/hotword path conditions transcription on domain terms, which is the task-specific adaptation beyond plain ASR. See [audio encoders](roadmap.md#vision-and-audio-encoders).

### Niche mechanism
The Whisper encoder (32L/1280, 128 mel, 3,000 frames) is followed by a stride-2 downsampler, then a Ling decoder with 128-expert top-8 MoE and MLA compressed KV (`q_lora` 256, `kv_lora` 512). Hotwords are prepended as text, so the decoder conditions on domain terms; one MTP layer adds a multi-token-prediction head. **Design takeaway:** pair a fixed acoustic encoder with a strong text decoder, and handle biasing by prompt rather than a separate hotword model.

### Training and task tailoring
The EU AI Act disclosure reports more than 10T text tokens and more than 1M hours of audio, with strong English/Chinese coverage and a cutoff around May 2026. SFT and RL stages are not stated. Context and hotword ASR are handled as prompt-conditioned transcription.

### How it is used
Use `AutoProcessor` + `AutoModelForCausalLM(trust_remote_code=True)`, BF16, SDPA, greedy decoding, 256 new tokens, no-repeat-ngram 10, and 16 kHz mono audio; call `prepare_asr_request`/`decode_transcript`. SGLang-Omni serves `/v1/audio/transcriptions` with `zh`/`en` hints, hotwords, and SSE streaming.

### Fit and caveats
The tag fits, but it needs custom code; it is single-turn, Chinese/English only, and produces no timestamps or diarization. The bundled example is non-streaming, streaming is server-side, hotwords are hints, and long audio needs chunking. Reported error rates are 4.41% overall, 1.67% domain, 8.42% dialect, 3.55% Chinese, and 3.99% English, versus Qwen3-ASR-1.7B at 6.52/2.49/15.30/3.66/4.63 and Fun-ASR 1.5 at 5.17/4.14/8.00/4.24/4.30.

## Related Reading
- [Roadmap: audio encoders](roadmap.md#vision-and-audio-encoders) and [post-training](roadmap.md#post-training-sft-preferences-and-rl).
- [Audio-Text-to-Text](audio-text-to-text.md) · [Audio-to-Audio](audio-to-audio.md) · [Voice Activity Detection](voice-activity-detection.md)
