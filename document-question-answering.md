# Document Question Answering (`document-question-answering`)

## Overview
Document Question Answering focuses on extracting or generating precise answers to user queries directed at visually rich documents. Unlike standard text question answering, this task processes complex visual layouts such as scanned PDF contracts, receipts, financial tables, forms, and diagrams. Models designed for this domain must interpret typography, bounding boxes, and structural alignments alongside raw alphanumeric characters. By merging optical character recognition with layout geometry and language understanding, these systems reliably locate specific numerical figures, verify clause terms, and aggregate tabular data. They are widely applied in financial auditing, automated invoice processing, legal compliance review, and enterprise document search workflows.

## Representative Model: `eachadea/ggml-vicuna-13b-1.1`

### Model profile
This repository is a 2023 community quantization of Vicuna-13B-v1.1 in the legacy GGML format. It contains text-only 13B weights in several q4/q5/q8 variants (plus "uncensored" files), with no safetensors, config, tokenizer, image tower, or OCR front-end. The card itself says "Obsolete model." It is a historical artifact, not a document-understanding system.

### Architecture and training
The underlying model is LLaMA-13B (40 layers, 5120 hidden, 40 heads) instruction-tuned on roughly 70K ShareGPT conversations for about $300. There is no visual encoder, layout model, or document-QA objective, and no RLHF. The only "tailoring" is Vicuna's chat format and general instruction following; nothing in training targets scanned pages, tables, forms, or spatial reasoning.

### Niche mechanism
There is none for document QA: LLaMA-13B is a 40L/5120/40-head causal decoder with no image encoder, OCR, layout embedding, or spatial attention. Its only conditioning is a chat prompt. A real document-QA stack needs a page encoder (ViT or OCR+layout transformer) and a fusion path into the language model. **Design takeaway:** text-only LLMs cannot answer layout questions; add a page-image encoder and explicit spatial tokens.

### How it is used
The original loader is old `llama.cpp` with a GGML `.bin` file such as `ggml-vic13b-q4_0.bin`, using Vicuna's `USER:`/`ASSISTANT:` prompt format. Current llama.cpp expects GGUF, so conversion or an old runtime is required. There is no Transformers pipeline.

### Fit and caveats
This is a severe tag mismatch: the checkpoint is a text-only chatbot and cannot see a document at all, let alone answer layout-aware questions. The repo also has no document-QA metrics or benchmark evidence. It is included here because the Hub maps the `document-question-answering` task to it, but it should not be used for that task.

## Related Reading
- [Roadmap: task-tag caveats](roadmap.md#task-tag-caveats) and [masked vs. autoregressive objectives](roadmap.md#masked-vs-autoregressive-objectives).
- [Visual Question Answering](visual-question-answering.md) · [Image-to-Text](image-to-text.md) · [Visual Document Retrieval](visual-document-retrieval.md)
