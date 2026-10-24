# Table Question Answering (`table-question-answering`)

## Overview
Table Question Answering systems query structured and semi-structured tabular data in response to natural language inquiries. Unlike standard text processors, these models incorporate two-dimensional spatial position embeddings corresponding to rows and columns, enabling them to understand relational tabular geometry. The model performs complex reasoning operations including row filtering, column selection, numerical comparison, and mathematical aggregation such as sums or averages. Table QA is vital for financial spreadsheet analysis, business intelligence dashboards, database natural language querying, and automating reporting workflows across corporate accounting.

## Representative Model: `AIAT/Kiddee-qatable1`

### Model profile
Kiddee-qatable1 is a 13.11B-parameter F16 chat model (about 26.2 GB) created in May 2024 under Apache-2.0, targeting Thai and English. It is a Llama-2-13B-geometry model (40 layers, 5120 hidden, 40 Q/40 KV heads, 13,824 FFN, 4,096-token context, 41,088 vocabulary, RoPE theta 10,000). Tables are serialized into the prompt; there is no TAPAS-style table encoder or cell-coordinate output.

### Architecture and task limitations
The model is a plain causal language model, so its "table understanding" depends entirely on how rows and columns are flattened into text. It cannot use two-dimensional position embeddings, and it cannot return cell coordinates. The card describes a retrieval-augmented QA system rather than a table-specific architecture.

### Niche mechanism
There is no table encoder: Llama-2-13B has 40 Q/40 KV heads and a 4K context, and the table is flattened into the prompt. Any table reasoning is therefore text reasoning, with no 2-D position embeddings and no cell-coordinate head. **Design takeaway:** real table QA needs either a table encoder (TAPAS-style) or a parser/tool (text-to-SQL/Pandas) plus schema linking; serialization alone is brittle.

### Training and task tailoring
The card gives no training or table-preprocessing details. A linked `AIAT/Kiddee-data1234` dataset has 24,376 Text→Pandas rows (Thai/English, Gemini-Pro generated), and its collection links to a Thai Table QA Hackathon submission, so text-to-Pandas tuning is likely but undocumented. The repository ships full F16 weights with no adapter config despite the `adapter-transformers` tag.

### How it is used
Load it as a causal chat model with `AutoModelForCausalLM` and serialize the table plus question into a 4K-token prompt, expecting pandas/Python or an answer. Do not use the TAPAS-style `table-question-answering` pipeline, which expects a table encoder.

### Fit and caveats
The tag overstates documented support. There is no table encoder, no table-specific evaluation, and the reported accuracy of 0.53 and response time of 2.440 have unstated units and protocol. The likely text-to-Pandas training may work for small tables, but long or complex tables will hit the 4K context limit.

## Related Reading
- [Roadmap: task-tag caveats](roadmap.md#task-tag-caveats) and [masked vs. autoregressive objectives](roadmap.md#masked-vs-autoregressive-objectives).
- [Tabular Classification](tabular-classification.md) · [Question Answering](question-answering.md) · [Summarization](summarization.md)
