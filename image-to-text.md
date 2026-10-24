# Image-to-Text (`image-to-text`)

## Overview
Image-to-Text models inspect an input image and generate corresponding natural language text. This covers several distinct applications including automated image captioning, optical character recognition for extracting printed and handwritten text, and visual document transcription. The model combines convolutional or vision transformer encoders with autoregressive language decoders to translate visual features into coherent textual tokens. This task is crucial for web accessibility by providing automated screen reader descriptions for visually impaired users, digitizing historical paper archives, automating invoice processing, and generating descriptive metadata for multimedia search engines.

## Representative Model: `numind/NuExtract3`

### Model profile
NuExtract3 is a 4.54B BF16 vision-language model (about 9.3 GB) released in May 2026 under Apache-2.0. It accepts text, images, or interleaved text+images and returns schema-constrained JSON or Markdown/HTML/LaTeX content. It is an extraction model, not a free-form captioner.

### Architecture tailored to structured extraction
It is built on Qwen3.5-4B: 32 text layers, 2560 hidden, three linear-attention layers per full-attention layer, 262K context, and a 24-layer ViT. Its chat template exposes `structured`, `content`/`markdown`, `template-generation`, `instructions`, in-context examples, and `enable_thinking`. This prompt-level control is the task adaptation: a JSON template defines the output schema, and the model fills it from the document. See [Qwen family](roadmap.md#model-family-notes).

### Niche mechanism
NuExtract3 does not add a special extraction head: it uses the Qwen3.5-4B VLM's 3:1 Gated DeltaNet/full-attention stack and a 24-layer ViT, then constrains generation with chat-template fields for schema, instructions, ICL examples, and thinking. SFT + RL teach it to fill the template rather than caption. **Design takeaway:** structured extraction is a prompt/decoding contract, not a new architecture; train the model to obey the schema and validate the JSON.

### Training and task tailoring
Starting from Qwen3.5-4B, NuExtract3 is SFT-trained with next-token prediction and then RL-trained for template adherence and extraction reasoning. Data mixes real Fine-PDF documents with complex synthetic documents. Thinking traces are trained to roughly answer length and can be switched off for latency.

### How it is used
Run with Transformers or vLLM (`--max-model-len 131072`), optionally with `qwen3_next_mtp` speculative decoding (2 tokens). Pass `template`, `instructions`, `mode`, and `enable_thinking` as chat-template kwargs; use temperature 0.2 in non-thinking mode and 0.6 in thinking mode.

### Fit and caveats
The tag is `image-to-text`, but the output is schema-constrained JSON or Markdown/HTML, not a description. Reported internal results over ~600 documents are 0.651 average versus 0.538 for Gemma-4-E4B-it, 0.479 for Qwen3.5-9B, and 0.417 for Qwen3.5-4B; thinking mode reaches 0.701, and document-to-Markdown scores 70.1. The benchmark is internal, and 27 outputs failed deserialization, so validate schemas in production.

## Related Reading
- [Roadmap: multimodal fusion](roadmap.md#multimodal-fusion) and [Qwen family](roadmap.md#model-family-notes).
- [Document Question Answering](document-question-answering.md) · [Visual Question Answering](visual-question-answering.md) · [Visual Document Retrieval](visual-document-retrieval.md)
