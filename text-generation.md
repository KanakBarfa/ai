# Text Generation (`text-generation`)

## Overview
Text Generation models autoregressively predict subsequent tokens based on preceding textual context to synthesize coherent, contextually relevant text passages. Utilizing decoder-only transformer architectures trained on vast internet-scale text corpora, these models excel across an enormous range of tasks including code completion, conversational dialogue, creative storytelling, technical documentation, and complex multi-step reasoning. Modern text generation systems incorporate techniques like grouped-query attention, rotary embeddings, and mixture-of-experts architectures to achieve state-of-the-art fluency and compute efficiency. They power modern generative AI assistants, automated programming tools, customer communication agents, and analytical problem-solving pipelines worldwide.

## Representative Model: `reflection/Beam-1.0-501B-A23B`

### Model profile
Beam-1.0 is a 501.27B-parameter (23.4B active) sparse MoE language model released in October 2026 under Apache-2.0. The BF16 repository is about 1 TB, with FP8 and NVFP4 variants available. It is English-focused, supports a 1M-token context, and is instruction-tuned for coding, agentic, and STEM work rather than being a raw base model.

### Architecture tailored to long-context generation
The stack has 52 layers (3 dense plus 49 MoE), 5120 hidden, 48 Q/8 KV heads with head_dim 128, 256 experts with top-8 routing, moe_intermediate 2560, and a 151,936 vocabulary. It uses gated GQA with 512-token sliding/full attention, softcap 30, and routed scaling 2.5, plus a modified Harmony template. The sparse MoE plus sliding attention is what makes 1M context and 23B active compute practical. See [mixture-of-experts](roadmap.md#mixture-of-experts).

### Niche mechanism
Beam mixes 512-token sliding-window and full attention, so local layers keep a bounded KV cache while global layers retain long-range recall. Gated GQA adds an output gate; logit softcap 30 bounds attention scores; routed scaling 2.5 keeps the 256-expert top-8 MoE stable; three dense layers warm up representations before 49 MoE layers. **Design takeaway:** long context is a KV-cache problem—hybrid local/global attention + sparse MoE + bounded logits is the recipe.

### Training and task tailoring
Pretraining uses public/licensed text (web, code, docs, educational/scientific), followed by midtraining on curated reasoning, tool-use, and long-context data. Post-training combines RL on coding/agentic/STEM environments with instruction tuning. The card reports filtering, deduplication, and decontamination; the cutoff is 30 June 2026.

### How it is used
Serve with a Harmony-aware engine. Reasoning effort is selectable at 8/32/64/128/256/512 tokens (default 512), and tools use a flat TypeScript schema. BF16/FP8/NVFP4 are supported. The card gives no runnable code sample.

### Fit and caveats
The tag fits, but this is an instruction-tuned agentic model, not a base generator. Reported BF16/FP8/NVFP4 scores include SWE-bench Verified 81.6/80.2/80.7, SWE Pro v1 66.1/65.9/65.6, Terminal Bench v2.1 80.1/78.9/78.4, AIME 2026 97.7/98.0/97.5, GPQA Diamond 90.6/90.3/89.3, HLE 36.0/51.1, and MCP Atlas 80.8. All are self-reported at temperature 0.7/top_p 0.99, and the technical report is pending.

## Related Reading
- [Roadmap: mixture-of-experts](roadmap.md#mixture-of-experts) and [post-training](roadmap.md#post-training-sft-preferences-and-rl).
- [Translation](translation.md) · [Reinforcement Learning](reinforcement-learning.md) · [Any-to-Any](any-to-any.md)
