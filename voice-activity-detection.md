# Voice Activity Detection (`voice-activity-detection`)

## Overview
Voice Activity Detection, or VAD, identifies the exact boundaries where human speech begins and ends within a continuous audio stream, distinguishing spoken voice from ambient noise and silence. Operating with extremely low latency, VAD models act as an energy-efficient gatekeeper in voice processing pipelines, ensuring heavy downstream ASR or translation models only activate when human speech is genuinely present. This filtering preserves privacy and drastically reduces server compute costs and battery consumption across smart speakers, mobile devices, VoIP communication tools, teleconference meeting recorders, and digital hearing aids.

## Representative Model: `nvidia/Nemotron-3-Diarization`

### Model profile
Nemotron-3-Diarization is a 99.2M-parameter speech model released in September 2026 under OpenMDW 1.1, with F32 safetensors, a `.nemo` file, and a q8_0 GGUF (~107 MB). It takes 16 kHz mono audio and returns `[T, 8]` per-speaker activity probabilities at 10 ms resolution or `(start, end, speaker)` segments. Despite the `voice-activity-detection` tag, it is an 8-speaker diarizer, not a binary VAD.

### Architecture tailored to diarization
A 31-layer transformer encoder (512 hidden, 8 heads, RoPE, 128 mel) processes features; 10 ms features are stacked 8× to 80 ms and a Conv1D maps back to 10 ms. An arrival-order speaker cache (AOSC) plus FIFO maintains speaker identity across time, with 8 arrival-ordered channels. This streaming sortformer design is what allows online speaker tracking rather than simple speech/silence detection. See [audio encoders](roadmap.md#vision-and-audio-encoders).

### Niche mechanism
Sortformer is a streaming diarizer: 31 transformer layers process 80 ms stacked features and a Conv1D upsamples back to 10 ms, while an arrival-order speaker cache (AOSC) plus FIFO keeps a fixed set of 8 speaker slots. The output is `[T,8]` per-frame activity, so it tracks who speaks when rather than just speech/silence. **Design takeaway:** streaming multi-speaker VAD needs a speaker cache and fixed slots; a binary VAD head loses identity.

### Training and task tailoring
It uses NEST self-supervised initialization, then two stages on 8×8 A100-SXM4-80GB: offline training on simulated mixtures, followed by streaming fine-tuning on real plus simulated data. Data includes about 10,000 h of real conversations (Fisher, AMI, ICSI, AISHELL-4, CALLHOME, NOTSOFAR1, DISPLACE, YODAS-v2, etc.) and 82,611 h of FastMSS-simulated 1–8-speaker mixtures.

### How it is used
Use NeMo `SortformerEncLabelModel.from_pretrained(...)` with `chunk_len`, `chunk_right_context`, `fifo_len`, and `spkcache_update_period`, then `diarize()`; or use Transformers `AutoModelForAudioFrameClassification`; NeMo-Speech.cpp is also supported. BF16 is typical, with reported latencies of 30.4/1.04/0.64/0.32 s.

### Fit and caveats
The tag is a mismatch: the output is frame-level multi-speaker diarization, not VAD. The 8-speaker cap is hard, DER depends on the reference RTTM/forced-alignment protocol, and streaming adds latency. Reported DER is 12.73 on DIHARD III (vs 19.09), 9.10 on CALLHOME, 11.14 on AMI SDM, and 6.77 on NOTSOFAR1.

## Related Reading
- [Roadmap: audio encoders](roadmap.md#vision-and-audio-encoders) and [task-tag caveats](roadmap.md#task-tag-caveats).
- [Automatic Speech Recognition](automatic-speech-recognition.md) · [Audio-to-Audio](audio-to-audio.md) · [Audio Classification](audio-classification.md)
