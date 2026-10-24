# Summarization (`summarization`)

## Overview
Summarization condenses extensive written documents into concise overviews containing the most critical facts and narrative themes. Extractive summarization identifies and stitches together key source sentences, while abstractive summarization synthesizes fluent, original phrasing to distill core concepts. Modern summarization models leverage long-context transformer architectures with sparse attention mechanisms to process entire research papers, legal contracts, or books. Summarization is deployed across news aggregation platforms, corporate briefing tools, executive dashboard generators, and legal discovery workflows to save reading time and highlight key findings.

## Representative Model: `mradermacher/Compactor-Qwen3.5-4B-GGUF`

### Model profile
Compactor is a 4.21B-parameter Qwen3.5-4B fine-tune distributed as 12 static GGUF quantizations from Q2_K (2.0 GB) to f16 (8.5 GB), with Q4_K_M (2.8 GB) recommended. It was quantized by mradermacher from `schneewolflabs/Compactor-Qwen3.5-4B` and released in September 2026 under Apache-2.0. It is a fixed-format context-compaction component, not a general summarizer.

### Architecture and task tailoring
The base is Qwen3.5-4B: 32 layers, 2560 hidden, eight blocks of three Gated DeltaNet layers plus one Gated Attention layer, 16 Q/4 KV heads, head_dim 256, and 262K context. The SFT adapter was merged into the weights, so there is no PEFT dependency. The task is to summarize an agent conversation that is about to be dropped, preserving the task, findings, and state for the `egirl` agent. See [Qwen family](roadmap.md#model-family-notes).

### Niche mechanism
The base is Qwen3.5-4B with 24 Gated DeltaNet layers and 8 full-attention layers (3:1), so 262K context is cheap, but the task is narrow: LoRA r=64/alpha=128 was merged and trained on 1,059 agent windows to emit a fixed compaction schema. Thinking is disabled at inference to match the pre-closed training template. **Design takeaway:** a small long-context model + schema-specific SFT can replace a generic summarizer for one workflow.

### Training and task tailoring
The base was `Lazarus-Ai/ReAligned-Qwen3.5-4B`. SFT used LoRA r=64, alpha=128, merged into the weights, on the `egirl-compaction-SFT` dataset: 1,059 train and 45 held-out windows, 9.34M tokens, one epoch, 133 steps, lr 1e-4 cosine, sequence 16,384, 11h44m on one RTX A6000. Targets are GPT-5 summaries.

### How it is used
Run with llama.cpp, Ollama, or Transformers; render with `enable_thinking=False` because training used a pre-closed think block. Prompt with the conversation about to be dropped and expect a structured summary. Q4_K_M is the recommended balance.

### Fit and caveats
The tag is only approximate. This is English-only, trained on one operator's 120 agentic-coding transcripts for one epoch. On 45 held-out windows, the Task line went from 0/45 to 45/45, task overlap word F1 from 0.000 to 0.509, fact recall from 0.157 to 0.661, and compression from 0.043 to 0.129; mean output grew from 884 to 2,689 characters. Do not use it as a general document summarizer.

## Related Reading
- [Roadmap: adapters, LoRA, and quantization](roadmap.md#adapters-lora-and-quantization) and [Qwen family](roadmap.md#model-family-notes).
- [Text Generation](text-generation.md) · [Question Answering](question-answering.md) · [Feature Extraction](feature-extraction.md)
