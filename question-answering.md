# Question Answering (`question-answering`)

## Overview
Question Answering systems accept a user question paired with context passages or query an external knowledge repository to return an accurate answer. Extractive question answering models locate and highlight the exact start and end span of text within the provided passage that answers the query. Abstractive models synthesize free-form answers using internal reasoning capabilities. Question Answering architectures are foundational to automated customer support assistants, enterprise search portals, clinical decision support tools, and modern retrieval augmented generation pipelines where accurate factual grounding is essential.

## Representative Model: `Jackrong/Trace-Inverter-4B`

### Model profile
Trace-Inverter-4B is a 4.15B-parameter BF16 model (about 8.3 GB) released in May 2026 under Apache-2.0. It supports six languages and is built on Qwen3-4B-Instruct-2507 (36 layers, 2560 hidden, 32 Q/8 KV heads, 262K context). Despite the `question-answering` tag, it is an answer-conditioned reasoning-trace reconstructor, not an extractive or retrieval QA model.

### Architecture and task definition
The architecture is the standard Qwen3 decoder with a chat template that supports `<think>` blocks. The task is: given a problem, a supplied final answer, and compressed "reasoning bubbles," reconstruct the full synthetic reasoning trace that leads to that answer. The model's job is therefore trace inversion—generating plausible intermediate steps conditioned on an answer—rather than finding evidence or answering from context. See [text generation](text-generation.md).

### Niche mechanism
There is no retrieval or span head: Qwen3-4B is a standard 36L/2560/32Q-8KV decoder, and SFT teaches it to generate a full `<think>` trace conditioned on the problem, a supplied answer, and compressed bubbles. That is answer-conditioned trace synthesis, not evidence grounding. **Design takeaway:** QA needs a retrieval/evidence path and a verifier; a generator trained to justify a given answer will rationalize errors.

### Training and task tailoring
Training is supervised fine-tuning only. Inputs are problem + supplied final answer + compressed bubbles; targets are full synthetic traces. Source traces come from GLM-5.1 and DeepSeek-v4, compressed by Qwen3-235B-A22B-Instruct-2507 and filtered for completeness and consistency. The card calls the dataset a small early prototype.

### How it is used
Run with Transformers or vLLM in BF16. Use the documented system prompt and the Problem / Model's final answer / Reasoning Bubbles fields; output is wrapped in `<think>...</think>`. It needs 262K context, and occasional `</tool_call>` artifacts should be stripped.

### Fit and caveats
This is a task mismatch: the model does not answer questions over evidence. It can rationalize a wrong supplied answer and produce synthetic rather than faithful chain-of-thought, and it is prone to templating. There is no quantitative benchmark or win rate—only a qualitative pizza arithmetic example.

## Related Reading
- [Roadmap: task-tag caveats](roadmap.md#task-tag-caveats) and [post-training](roadmap.md#post-training-sft-preferences-and-rl).
- [Text Generation](text-generation.md) · [Summarization](summarization.md) · [Token Classification](token-classification.md)
