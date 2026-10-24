# Roadmap: How to Read the 52 Task Categories

This roadmap is the shared reference for the whole catalogue. Read it once, then use the task files. Each task file keeps its ~100-word overview and adds a compact spec: exact architecture, a niche-mechanism note with a design takeaway, training, use, and caveats. Anything that would otherwise be repeated across many files (transformer anatomy, linear attention, diffusion, MoE, quantization, post-training, evaluation caveats) is defined here instead.

The order below is dependency-based, not alphabetical. It moves from representations to perception, then generation, then embodied and structured-data tasks, and finishes with the omni-modal capstone. If you only need one task, read the relevant foundation entries and jump straight to that file.

## Shared foundations

### Transformers and attention

Almost every model here is a transformer. A decoder-only stack predicts the next token with causal attention; an encoder (BERT-style) sees both directions; a cross-encoder sees a query and document together. Multi-query and grouped-query attention (GQA) share key/value heads to cut KV-cache memory, while rotary position embeddings (RoPE) encode positions. Several 2026 models replace most full-attention layers with linear-attention layers: Qwen3.5/3.8 uses three Gated DeltaNet layers followed by one full-attention layer, which makes 262K-token contexts practical. Sliding-window and hybrid attention recur in translation, diarization, and long-context models. See [linear attention and Gated DeltaNet](#linear-attention-and-gated-deltanet).

Used by: nearly all 52 files.

### Tokenization and embeddings

Text is split into subword tokens; images into patches; audio into frames or codec tokens; video into spatiotemporal tokens. An embedding layer maps these units into vectors, and a pooling strategy (CLS, mean, learned query) turns a sequence into one vector. Contrastive embedding models are trained so that a query and its matching document land close together in cosine space; Matryoshka training lets one embedding be truncated to 64–2048 dimensions with graceful accuracy loss. This distinction—generative logits versus a single embedding—is the main fork between `feature-extraction`/`sentence-similarity` and `text-generation`.

Used by: [feature-extraction](feature-extraction.md), [sentence-similarity](sentence-similarity.md), [text-ranking](text-ranking.md), [image-feature-extraction](image-feature-extraction.md), [visual-document-retrieval](visual-document-retrieval.md), [zero-shot-image-classification](zero-shot-image-classification.md).

### Vision and audio encoders

Vision Transformers (ViTs) split an image into 14–16 px patches and attend over them. Variants add register tokens, QK-Norm, SwiGLU, windowed attention, or deepstack features that are injected into a language model at several depths. Audio encoders are usually convolutional front-ends plus transformers: wav2vec 2.0 for self-supervised speech, Whisper-style encoders for ASR, and AuT for omni-modal audio. Audio codecs (Mimi, Code2Wav, visual/audio VAEs) turn continuous signals into discrete or latent tokens so a transformer can generate them.

Used by: [image-classification](image-classification.md), [image-segmentation](image-segmentation.md), [object-detection](object-detection.md), [automatic-speech-recognition](automatic-speech-recognition.md), [text-to-speech](text-to-speech.md), [text-to-audio](text-to-audio.md).

### Multimodal fusion

There are three common patterns. A projector maps vision/audio encoder outputs into the language model's token space (LLaVA-style). Cross-attention lets a language decoder query frozen visual features (BLIP-2-style). A unified token stream packs text, image, audio, and video tokens into one sequence and runs a single transformer over all of them (Qwen3-Omni, MiniMax H3). Fusion choice determines whether a model can only understand media or also generate it. Thinker–Talker designs separate reasoning from speech synthesis.

Used by: [visual-question-answering](visual-question-answering.md), [image-text-to-text](image-text-to-text.md), [audio-text-to-text](audio-text-to-text.md), [video-text-to-text](video-text-to-text.md), [any-to-any](any-to-any.md), [image-text-to-video](image-text-to-video.md).

### Mixture-of-experts

A sparse MoE replaces one feed-forward network with many experts and a router that activates only the top-k per token. This raises total parameters without raising per-token compute: Qwen3.5-35B-A3B, Beam-501B-A23B, North-218B-A25B, and Drama-3-35B-A3B all use this idea. Routers may be sigmoid or softmax, with shared experts, expert-choice routing, or auxiliary-loss-free balancing. MoE makes frontier models cheaper to serve but harder to quantize and fine-tune.

Used by: [text-generation](text-generation.md), [translation](translation.md), [text-to-speech](text-to-speech.md), [reinforcement-learning](reinforcement-learning.md), [audio-text-to-text](audio-text-to-text.md).

### Diffusion, flow matching, and VAEs

Diffusion models learn to reverse a noising process; flow-matching models learn a velocity field between noise and data. In latent diffusion, a VAE compresses images/video/audio into a smaller latent space, a denoiser (U-Net, DiT, or MMDiT) operates there, and the VAE decodes the result. MMDiT keeps text and image tokens in separate streams with joint attention. Distillation reduces sampling to 1–8 steps (DMD2, trajectory distribution matching), which is how Turbo and 8-step video models work. Classifier-free guidance trades diversity for prompt adherence.

Used by: [unconditional-image-generation](unconditional-image-generation.md), [text-to-image](text-to-image.md), [image-to-image](image-to-image.md), [image-text-to-image](image-text-to-image.md), [text-to-video](text-to-video.md), [image-to-video](image-to-video.md), [video-to-video](video-to-video.md), [image-to-3d](image-to-3d.md).

### Masked vs. autoregressive objectives

Masked language modeling (MLM) hides tokens and predicts them from bidirectional context; it is the classic pretraining task behind BERT-family encoders and ESM-2 protein models. Autoregressive next-token prediction powers decoder-only generation. Span denoising, prefix-LM, and fill-in-the-middle objectives sit between the two. The objective determines whether a checkpoint is naturally a classifier/embedder or a generator.

Used by: [fill-mask](fill-mask.md), [text-generation](text-generation.md), [image-to-text](image-to-text.md), [translation](translation.md).

### Contrastive and metric learning

Contrastive training pulls matching pairs together and pushes negatives apart (InfoNCE). Bi-encoders embed query and document independently and are fast for retrieval; cross-encoders attend over both jointly and are more accurate but slower, so they are used for reranking. Hard-negative mining and synthetic queries are what make modern retrieval models work. Cosine similarity, dot product, and calibrated logits are the usual outputs.

Used by: [sentence-similarity](sentence-similarity.md), [text-ranking](text-ranking.md), [visual-document-retrieval](visual-document-retrieval.md), [image-feature-extraction](image-feature-extraction.md), [video-classification](video-classification.md).

### Adapters, LoRA, and quantization

LoRA freezes the base model and trains low-rank update matrices. IC-LoRA adds a reference-conditioning path so the adapter can read an input video while denoising. GGUF is a llama.cpp file format with many quantization levels (Q2_K to Q8_K_XL); FP8/FP4 are hardware-friendly datatypes for serving large models. Distillation compresses a teacher's sampling trajectory or outputs into a smaller/faster student. Always check whether a repository contains full weights, a LoRA adapter, a quantized derivative, or only a projection/encoder component.

Used by: [image-to-video](image-to-video.md), [video-to-video](video-to-video.md), [feature-extraction](feature-extraction.md), [summarization](summarization.md), [image-text-to-image](image-text-to-image.md).

### Post-training: SFT, preferences, and RL

Supervised fine-tuning (SFT) teaches format and task behavior from demonstrations. Preference optimization (DPO, STPO) aligns outputs with human or model preferences. Online RL (PPO, GRPO, GSPO) optimizes verifiable rewards or learned reward models; RLVR uses executable checkers. Some systems add self-improvement loops where the model proposes its own training tasks or harness changes. These stages are often what turn a generic backbone into a task specialist—and they are also why benchmark claims can be hard to compare.

Used by: [reinforcement-learning](reinforcement-learning.md), [robotics](robotics.md), [text-generation](text-generation.md), [translation](translation.md), [text-to-image](text-to-image.md), [text-to-speech](text-to-speech.md), [automatic-speech-recognition](automatic-speech-recognition.md).

### Evaluation caveats

Most numbers in these files are vendor-reported. Protocols differ (greedy vs. sampled, LLM judges, internal test sets, test-tuned thresholds), and some cards mix base-model and checkpoint results. Treat cross-model comparisons as directional unless the same harness and split were used. The `pipeline_tag` on the Hub is also only a hint: several models in this catalogue are deliberately or accidentally tagged with a neighbouring task, and each file calls that out.

## Mechanisms from scratch

These are the primitives the model files assume. Each note is implementation-level: what the tensors do, why the trick exists, and what it costs. Read the one you need; the model files only describe the task-specific twist.

### Attention, RoPE, and the KV cache

A transformer block projects each token to query, key, and value vectors, computes `softmax(QK^T / sqrt(d_head))`, and mixes values. Causal attention masks future tokens. Multi-head attention (MHA) gives every head its own Q/K/V; multi-query attention (MQA) shares one K/V head across all query heads; grouped-query attention (GQA) shares one K/V head per group of query heads. GQA is the usual 2026 compromise: near-MHA quality with a much smaller KV cache.

KV-cache bytes are `2 × layers × kv_heads × head_dim × tokens × bytes_per_element`. That formula, not parameter count, decides long-context serving cost. RoPE rotates each Q/K pair by an angle proportional to position, so the dot product depends on relative offset. Partial RoPE rotates only a fraction of the dimensions. YaRN rescales RoPE frequencies to extend context. QK-norm applies RMSNorm to Q and K before the dot product; it stabilizes large models and low-precision training. Logit softcapping computes `c·tanh(scores/c)`, bounding attention logits. A sliding-window layer attends only to the last `W` tokens; interleaving it with full-attention layers gives cheap local mixing plus periodic global recall.

### Linear attention and Gated DeltaNet

Softmax attention costs `O(n²)` time and an `O(n)` KV cache. Linear attention replaces the softmax kernel with a feature map `φ` and maintains a running state:

`S_t = S_{t-1} + φ(k_t) v_t^T`, `y_t = φ(q_t)^T S_t / (φ(q_t)^T z_t)`, with `z_t = z_{t-1} + φ(k_t)`.

State size is `d_k × d_v` per head, independent of sequence length. Training uses a chunkwise parallel form (matrix multiplies over chunks); inference is a recurrence with `O(1)` state per token. The cost is loss of exact softmax recall: a fixed-size state cannot copy arbitrary long-range details.

Gated DeltaNet adds a forget gate `α_t` and a write strength `β_t` to the delta rule:

`S_t = α_t S_{t-1} + β_t (v_t - α_t S_{t-1} k_t) k_t^T`.

Read this as: decay the old memory, compute the value the current key would retrieve, then write the residual `v_t - old`. It is an associative memory with targeted overwrite, not a plain sum. A short depthwise convolution before the Q/K/V projections supplies local n-gram context. Qwen3.5/3.8 interleaves three Gated DeltaNet layers with one full-attention layer; the full layer restores precise recall, while the linear layers make 262K context affordable.

### Mixture of experts

A router computes logits `r = W_r x`, selects the top-k experts, and combines their outputs with weights `g_i`. A shared expert can run for every token. Total parameters scale with the number of experts; per-token compute scales with top-k. Dropless training avoids token dropping; capacity factors trade load balance for waste. Aux-loss-free balancing updates a per-expert bias instead of adding an auxiliary loss that fights the language-model objective. Sigmoid routers, `sqrtsoftplus` routers, and expert-choice routing are stability/load-balancing variants, not different arithmetic. MoE saves serving FLOPs but makes quantization, fine-tuning, and multi-GPU serving harder because all experts must be resident somewhere.

### Normalization, activations, and positions

Pre-norm RMSNorm is the default: normalize the residual stream before attention and before the FFN. Zero-centered RMSNorm subtracts a learned mean from the scale. QK-norm stabilizes attention. SwiGLU uses `W2(SiLU(W1x) ⊙ W3x)` and is standard in modern LLMs; it costs three matrices instead of two. GeGLU is the same idea with GELU. Most 2026 models omit linear-layer biases.

Positions are usually RoPE, partial RoPE, or no explicit position at all in linear-attention layers. Multimodal models use 3D axial RoPE (time, height, width) or MRoPE (temporal, height, width) so image and video patches get meaningful coordinates. ALiBi is a simple additive distance bias; it is rare in frontier models but cheap.

### Vision and audio tokenization

A ViT splits an image into `P×P` patches, projects each patch to `d_model`, adds position, and runs transformer blocks. Register tokens are extra learned tokens that absorb high-norm global information; they improve dense features. Window attention limits cost to a local window, with global attention at selected blocks. Deepstack injects intermediate ViT features into several depths of a language model, not just the last layer. A Q-Former uses a small set of learned queries to cross-attend to frozen vision features, producing a fixed-length visual token set.

Audio has two common front ends. Wav2Vec-style models use a convolutional feature encoder over raw waveform, then a transformer, and are pretrained with masked-frame contrastive learning. Whisper-style models use log-mel spectrograms plus a transformer encoder. Neural audio codecs (Mimi, Code2Wav, RVQ-style) turn audio into several residual codebooks per frame; a depth transformer or per-codebook heads model those codebooks. Video adds a temporal axis: either 3D patches, frame stacking, or a Q-Former over sampled frames.

### Diffusion, flow matching, and VAEs

DDPM defines a forward noising chain `q(x_t|x_0)` and trains a network to predict the added noise `ε`: `L = E||ε - ε_θ(x_t,t)||²`. Sampling reverses the chain over 1,000 steps (or a faster sampler). The noise schedule is usually linear or cosine; `fixed_small` and `clip_sample` control the reverse variance and clipping.

Flow matching learns a velocity field along a path between noise `x_0` and data `x_1`, often the straight line `x_t = (1-t)x_0 + t x_1`; the target is `x_1 - x_0`, and an Euler solver integrates the field. Rectified flow is this straight-path objective. Classifier-free guidance (CFG) evaluates conditional and unconditional velocities and extrapolates: `v = v_uncond + s·(v_cond - v_uncond)`.

Latent diffusion compresses pixels with a VAE, denoises in the latent space, and decodes. DiT replaces the U-Net with a transformer; MMDiT keeps text and image tokens in separate streams with joint attention. AdaLN injects timestep and modality conditioning through per-block scale/shift. Distillation (DMD2, trajectory distribution matching) trains a student to match a teacher's trajectory in 1–8 steps; that is how “Turbo” and 8-step video models work.

### Heads and losses

The task determines the head. Language modeling uses a vocabulary projection and cross-entropy. Masked language modeling predicts masked tokens from bidirectional context. Contrastive learning uses InfoNCE on positive/negative pairs and cosine similarity. CTC marginalizes over alignments for speech. Token classification adds a per-token label head and often Viterbi decoding over valid label transitions. Detection uses DETR-style Hungarian matching with L1 + GIoU and a classification loss such as VFL. Segmentation uses cross-entropy/Dice; depth uses L1 plus gradient terms; forecasting uses pinball loss for quantiles; reward models use regression or Bradley–Terry.

### Build order from scratch

1. Tokenizer and data pipeline first; the vocabulary and sequence format constrain everything.
2. One pre-norm block: RMSNorm → GQA → residual → RMSNorm → SwiGLU → residual, with RoPE and QK-norm.
3. Choose the objective (CLM, MLM, contrastive, flow matching) and verify it overfits a small batch.
4. Add stability machinery only when needed: QK-norm, logit softcap, z-loss, u-μP, gradient clipping.
5. Add modality encoders/decoders as separate modules with explicit token interfaces.
6. Scale data and parameters; then post-train (SFT, preferences, RL) for behavior.
7. Evaluate with a fixed harness and held-out split; vendor numbers are not a substitute.

## Model family notes

Read the family once; the task files then only describe the task-specific adaptation.

| Family | What it gives you | Appears in |
|---|---|---|
| Qwen3.5/3.8 hybrid VLM | 27B-class backbone with 3:1 [Gated DeltaNet](roadmap.md#linear-attention-and-gated-deltanet)/full attention, 64 layers, 24 Q/4 KV heads, 27-layer ViT, 262K context; 35B-A3B variants add 256-expert MoE | [mammoth-1](text-classification.md), [d3-turbo](zero-shot-classification.md), [RoboRSI](robotics.md), [Flora-Rewarder](video-text-to-text.md), [NuExtract3](image-to-text.md), [Compactor](summarization.md), [Qwen3.8 Heretic](image-text-to-image.md), [Self-HPO](reinforcement-learning.md) |
| Qwen3.5-4B | 32-layer hybrid model, 2560 hidden, 16 Q/4 KV, 262K context; cheap extraction/summarization workhorse | [NuExtract3](image-to-text.md), [Compactor](summarization.md) |
| Qwen3-Omni | Thinker–Talker MoE with AuT audio encoder, 27-layer ViT, multi-codebook speech decoder | [any-to-any](any-to-any.md) |
| Mistral / Ministral 3 | 8B–24B dense models; Ministral-3-8B has 34 layers, 4096 hidden, 32 Q/8 KV; Pixtral vision tower | [Voxtral](audio-text-to-text.md), [Nemotron Embed](visual-document-retrieval.md), [Nemotron Rerank](text-ranking.md) |
| MMDiT image models | Joint text/image attention in a diffusion transformer; Qwen-Image-Edit (20.4B) and Krea 2 (12.8B) are separate implementations | [Qwen-Image-Edit](image-to-image.md), [Krea 2](text-to-image.md) |
| LTX-2.5 | 22B video DiT, Gemma 4 12B text encoder, video/audio VAEs, distilled 8-step path, LoRA/IC-LoRA adapters | [Cinemagraph](image-to-video.md), [Alpha-Gen](video-to-video.md), [LTX encoder](feature-extraction.md) |
| MiniMax H3 | 33B single-stream omni transformer, Qwen3-VL-32B encoder, visual/audio VAEs, synchronized audio-video | [MiniMax H3](image-text-to-video.md), [FastH3](text-to-video.md) |
| Sapiens / Sapiens2 | Human-centric ViT (1B/5B), MAE + contrastive pretraining, dense task heads for segmentation, depth, and pose | [Sapiens seg](image-segmentation.md), [Sapiens2 pointmap](depth-estimation.md), [Sapiens2 pose](keypoint-detection.md) |
| SAM3 / DINOv3 | Promptable concept segmentation (Perception Encoder + DETR + tracker) and self-supervised ViT-7B features | [SAM3](mask-generation.md), [DINOv3](image-feature-extraction.md) |
| Moshi / PersonaPlex | Full-duplex speech model with temporal/depth transformers over Mimi codec streams | [PersonaPlex](audio-to-audio.md) |
| FishOmni / Drama-3 | Qwen3.5 hybrid MoE plus a multi-codebook speech decoder | [Drama-3](text-to-speech.md) |
| ESM-2 | BERT-style protein masked-language model over a 33-token amino-acid vocabulary | [ESM-2](fill-mask.md) |
| Toto-2 | Patched decoder-only time-series transformer with a 9-quantile head | [Toto-2](time-series-forecasting.md) |
| Relational Transformer | Attention over linked database rows and cells, with text columns embedded by MiniLM | [RT-J](tabular-classification.md) |

## Reading order

### Tier 0 — Foundations

Read [Shared foundations](#shared-foundations) and [Mechanisms from scratch](#mechanisms-from-scratch), then skim [Model family notes](#model-family-notes). If you already know transformers and diffusion, use the mechanism primer as a refresher and the family table as a glossary.

### Tier 1 — Text representations and decisions

- [feature-extraction](feature-extraction.md) — how text becomes a vector.
- [sentence-similarity](sentence-similarity.md) — comparing two vectors.
- [fill-mask](fill-mask.md) — the masked objective behind encoders.
- [token-classification](token-classification.md) — per-token labels.
- [text-classification](text-classification.md) — whole-input labels.
- [zero-shot-classification](zero-shot-classification.md) — labels specified at inference time.
- [text-ranking](text-ranking.md) — cross-encoder reranking.

These build on [tokenization and embeddings](#tokenization-and-embeddings), [masked vs. autoregressive objectives](#masked-vs-autoregressive-objectives), and [contrastive learning](#contrastive-and-metric-learning).

### Tier 2 — Text generation and knowledge

- [text-generation](text-generation.md) — the decoder-only backbone and post-training.
- [question-answering](question-answering.md) — answering from supplied context.
- [summarization](summarization.md) — compression under a length budget.
- [translation](translation.md) — cross-lingual generation and evaluation.

Read [post-training](#post-training-sft-preferences-and-rl) before this tier.

### Tier 3 — Visual perception

- [image-classification](image-classification.md) — whole-image labels.
- [image-feature-extraction](image-feature-extraction.md) — visual embeddings.
- [object-detection](object-detection.md) — boxes for known classes.
- [zero-shot-object-detection](zero-shot-object-detection.md) — boxes from text.
- [image-segmentation](image-segmentation.md) — per-pixel labels.
- [mask-generation](mask-generation.md) — promptable masks.
- [keypoint-detection](keypoint-detection.md) — landmarks and pose.
- [depth-estimation](depth-estimation.md) — 3D geometry from 2D.
- [video-classification](video-classification.md) — clip-level labels and embeddings.

Read [vision and audio encoders](#vision-and-audio-encoders) first. Detection and segmentation assume the classification/embedding material.

### Tier 4 — Vision-language understanding

- [image-to-text](image-to-text.md) — captioning, OCR, and structured extraction.
- [visual-question-answering](visual-question-answering.md) — answering about an image.
- [document-question-answering](document-question-answering.md) — layout-aware document QA.
- [image-text-to-text](image-text-to-text.md) — general vision-language reasoning.
- [video-text-to-text](video-text-to-text.md) — video-language reasoning and reward scoring; note the tag mismatch in this catalogue.
- [zero-shot-image-classification](zero-shot-image-classification.md) — labels from text descriptions.
- [visual-document-retrieval](visual-document-retrieval.md) — embedding and retrieving document pages.

Read [multimodal fusion](#multimodal-fusion) before this tier.

### Tier 5 — Image generation and editing

- [unconditional-image-generation](unconditional-image-generation.md) — the diffusion baseline.
- [text-to-image](text-to-image.md) — prompt-conditioned synthesis.
- [image-to-image](image-to-image.md) — instruction-driven editing.
- [image-text-to-image](image-text-to-image.md) — multi-image/instruction editing; note the tag mismatch in this catalogue.

Read [diffusion, flow matching, and VAEs](#diffusion-flow-matching-and-vaes) first.

### Tier 6 — Video and 3D

- [text-to-video](text-to-video.md) — prompt-to-clip generation.
- [image-to-video](image-to-video.md) — animating a still.
- [image-text-to-video](image-text-to-video.md) — reference-conditioned audio-video generation.
- [video-to-video](video-to-video.md) — stylization, matting, and other video transforms.
- [text-to-3d](text-to-3d.md) — language-model mesh generation.
- [image-to-3d](image-to-3d.md) — image-to-mesh reconstruction.

Video generation reuses the diffusion foundations; 3D adds voxel/mesh representations and structured latents.

### Tier 7 — Audio and speech

- [audio-classification](audio-classification.md) — clip-level audio labels.
- [voice-activity-detection](voice-activity-detection.md) — speech activity and diarization.
- [automatic-speech-recognition](automatic-speech-recognition.md) — speech to text.
- [audio-to-audio](audio-to-audio.md) — speech-to-speech and enhancement.
- [text-to-speech](text-to-speech.md) — text to natural speech.
- [text-to-audio](text-to-audio.md) — music/sound description and generation; check direction in the file.
- [audio-text-to-text](audio-text-to-text.md) — joint audio-language reasoning.

Read [vision and audio encoders](#vision-and-audio-encoders) and [multimodal fusion](#multimodal-fusion) first.

### Tier 8 — Structured data

- [tabular-classification](tabular-classification.md) — labels from relational tables.
- [tabular-regression](tabular-regression.md) — continuous targets from flat tables.
- [table-question-answering](table-question-answering.md) — questions over tables.
- [time-series-forecasting](time-series-forecasting.md) — future values with quantiles.

These are deliberately late: they share little with the media stack and are best read as a separate applied track.

### Tier 9 — Decision-making and embodied models

- [reinforcement-learning](reinforcement-learning.md) — policies trained with rewards.
- [robotics](robotics.md) — planners and policies for physical agents.
- [graph-ml](graph-ml.md) — relational/graph data; note the tag mismatch in this catalogue.

Read [post-training](#post-training-sft-preferences-and-rl) and [transformers and attention](#transformers-and-attention) first.

### Tier 10 — Capstone

- [any-to-any](any-to-any.md) — a single model that combines perception, reasoning, and speech generation.

Read it last: it reuses the multimodal, audio, and generation foundations.

## Reading paths

- **Retrieval / RAG engineer:** Tier 1 → [feature-extraction](feature-extraction.md), [sentence-similarity](sentence-similarity.md), [text-ranking](text-ranking.md) → Tier 4 ([visual-document-retrieval](visual-document-retrieval.md), [document-question-answering](document-question-answering.md)).
- **Computer-vision engineer:** [vision and audio encoders](#vision-and-audio-encoders) → Tier 3 → Tier 4 → Tier 5.
- **Generative-media engineer:** [diffusion, flow matching, and VAEs](#diffusion-flow-matching-and-vaes) → Tier 5 → Tier 6 → [text-to-audio](text-to-audio.md).
- **Speech engineer:** [vision and audio encoders](#vision-and-audio-encoders) → [automatic-speech-recognition](automatic-speech-recognition.md) → [audio-to-audio](audio-to-audio.md) → [text-to-speech](text-to-speech.md) → [audio-text-to-text](audio-text-to-text.md).
- **Agent / robotics engineer:** [transformers and attention](#transformers-and-attention) + [post-training](#post-training-sft-preferences-and-rl) → [text-generation](text-generation.md) → [reinforcement-learning](reinforcement-learning.md) → [robotics](robotics.md) → [any-to-any](any-to-any.md).
- **Tabular / time-series analyst:** skip to Tier 8; read [tabular-regression](tabular-regression.md) before [tabular-classification](tabular-classification.md).

## Task-tag caveats

The Hub `pipeline_tag` is a discovery label, not a guarantee. In this catalogue:

- [document-question-answering](document-question-answering.md) points to an obsolete text-only Vicuna checkpoint.
- [image-text-to-image](image-text-to-image.md) points to a text-output VLM.
- [text-to-audio](text-to-audio.md) points to an audio captioner.
- [voice-activity-detection](voice-activity-detection.md) points to an 8-speaker diarizer.
- [video-text-to-text](video-text-to-text.md) points to a reward model.
- [graph-ml](graph-ml.md) points to a UI-code generator.
- [zero-shot-classification](zero-shot-classification.md) points to a fixed-code decision model.
- [feature-extraction](feature-extraction.md) points to an LTX conditioning encoder.

These are documented honestly in each file so the catalogue reflects what the checkpoint can actually do.
