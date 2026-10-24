# Audio-to-Audio (`audio-to-audio`)

## Overview
Audio-to-Audio encompasses tasks where an input audio waveform is processed to generate one or more modified output audio tracks. Prominent examples include speech enhancement to remove background noise, speech separation to isolate individual speakers talking simultaneously, audio style transfer, and voice conversion where a speaker's vocal timbre is translated to match another identity while retaining linguistic content. These architectures operate directly on raw waveforms or time-frequency representations. Primary use cases include teleconferencing audio cleanup, music remixing, studio mastering, and dubbing.

## Representative Model: `nvidia/personaplex-7b-v1`

### Model profile
PersonaPlex-7B-v1 is an 8.37B BF16 full-duplex speech model (one safetensors file, ~17.1 GB) released between December 2025 and March 2026. Weights use the NVIDIA Open Model License, code is MIT, and the Moshi base is CC-BY-4.0; access is gated. It takes streaming user speech plus a text role prompt and a short voice sample and returns streaming agent speech. It is English-only.

### Architecture tailored to duplex conversation
It is a Moshi duplex model: a temporal transformer over three streams (user audio, agent text, agent audio) plus a depth transformer for the audio codebooks, with the Mimi codec. The hybrid prompt concatenates text-role and voice segments, and the user channel is a 440 Hz sine with the loss masked. This design lets the model listen and speak simultaneously, which is the key difference from turn-based TTS/ASR pipelines. See [Moshi/PersonaPlex family](roadmap.md#model-family-notes).

### Niche mechanism
Mimi encodes 80 ms frames into eight residual codebooks. Moshi's temporal transformer models user audio, agent text, and agent audio as three parallel streams at 12.5 Hz; a depth transformer then predicts the eight codebooks for the current frame. During agent speech the user stream is a 440 Hz sine with loss masked, which trains full-duplex turn-taking instead of alternating turns. **Design takeaway:** model both speakers as parallel token streams with a codec, and mask the silent side rather than pausing it.

### Training and task tailoring
It starts from Moshi and trains on 1,840 h of service data (105,410 dialogs) plus 410 h of QA (39,322), using Adam at depth LR 4e-6 and temporal LR 2e-6 for 24,576 steps (batch 32, 2,048 tokens ≈ 163.84 s), about 6 h on 8×A100. The release adds 7,303 Fisher conversations (1,217 h); voices come from TortoiseTTS/ChatterboxTTS.

### How it is used
Accept the license, set `HF_TOKEN`, and install `moshi`. Run `moshi.server --ssl` or `moshi.offline --voice-prompt NATF2.pt` with the fixed NATF/NATM/VARF/VARM voices. BF16 is standard, with `--cpu-offload` available.

### Fit and caveats
`audio-to-audio` is broad; this is conversational duplex speech, not general audio transformation. Reported DMOS is 3.90 (Full-Duplex-Bench) and 3.59 (Service-Duplex-Bench) versus Gemini 3.72/3.22, speaker similarity 0.65 (Moshi 0.10), pause TOR 0.584 (Moshi 0.985), and turn-taking latency 0.070 (Moshi 0.265). It is gated, synthetic-heavy, and limited to fixed voices and English.

## Related Reading
- [Roadmap: Moshi/PersonaPlex family](roadmap.md#model-family-notes) and [audio encoders](roadmap.md#vision-and-audio-encoders).
- [Text-to-Speech](text-to-speech.md) · [Automatic Speech Recognition](automatic-speech-recognition.md) · [Audio-Text-to-Text](audio-text-to-text.md)
