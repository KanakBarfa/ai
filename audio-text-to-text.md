# Audio-Text-to-Text (`audio-text-to-text`)

## Overview
Audio-Text-to-Text models take multimodal inputs comprising spoken audio recordings paired with accompanying textual instructions or dialogue context, then output reasoned textual answers. Rather than acting merely as transcriptions, these models evaluate acoustic cues like intonation, emotion, speaker emphasis, and background noise in tandem with lexical meaning. This allows the system to comprehend conversational subtleties, follow complex spoken commands, answer questions about audio events, and engage in multi-turn voice-driven problem solving. Primary applications include interactive voice intelligence systems, real-time call center assistance, automated meeting summaries incorporating speaker intent, and auditory reasoning platforms for assistive technology.

## Representative Model: `mistralai/Voxtral-Small-24B-2507`

### Model profile
Voxtral Small is a 24.3B audio-language model released in July 2025 under Apache-2.0. The count is 640M audio encoder + 52M adapter + 670M text embeddings + 22.9B decoder. It accepts audio and text together and returns text, including multi-turn dialogue and tool calls. It is a transcription/audio-understanding model, not a speech generator.

### Architecture tailored to audio reasoning
A Whisper-style encoder (32 layers, 1280 hidden, 20 heads, 128 mel bins, 1,500 frames) produces audio features; an adapter injects them into a Mistral Small 3.1 decoder (40 layers, 5120 hidden, 32 Q/8 KV heads, 131K vocabulary, 32K effective context). The adapter is the task-specific bridge: it maps acoustic tokens into the same space as text so the language model can reason over prosody, content, and instructions in one sequence.

### Niche mechanism
Voxtral does not cross-attend: its 52M adapter projects Whisper-style encoder frames into the decoder's token space, so audio becomes a prefix of soft tokens in the same 40L Mistral sequence. Multiple clips are concatenated, and the 32K context covers roughly 30–40 minutes. **Design takeaway:** the cheapest audio-language bridge is a projector into a strong text decoder, not a new multimodal transformer.

### Training and task tailoring
The paper describes audio-text pretraining/alignment followed by instruction tuning for transcription and audio understanding; the card does not give a reproducible data mix or RL stage. The model is tuned for short-form transcription and audio QA in eight languages, with function calling preserved from the text backbone.

### How it is used
Recommended serving is vLLM 0.10+ with `--tokenizer_mode mistral --config_format mistral --load_format mistral` and tensor parallelism 2 (about 55 GB BF16), or Transformers 4.54+ with `mistral-common[audio]`. Use temperature 0 for transcription and 0.2/top_p 0.95 for chat. System prompts are not supported.

### Fit and caveats
The tag fits, but this is not a general audio-event reasoner. Reported WER is 1.53 on LS-Clean, 3.14 on LS-Other, 6.62 on VoxPopuli, and 3.35 on FLEURS-en, with vendor claims of SOTA English short-form ASR. Inputs are limited to roughly 30–40 minutes, and the Hub sets `inference: false` (no hosted widget).

## Related Reading
- [Roadmap: audio encoders](roadmap.md#vision-and-audio-encoders) and [multimodal fusion](roadmap.md#multimodal-fusion).
- [Automatic Speech Recognition](automatic-speech-recognition.md) · [Audio-to-Audio](audio-to-audio.md) · [Any-to-Any](any-to-any.md)
