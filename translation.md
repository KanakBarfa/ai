# Translation (`translation`)

## Overview
Translation converts written natural language text from one source human language into a target language while preserving the original meaning, style, and grammatical nuances. Modern machine translation architectures utilize sequence-to-sequence encoder-decoder transformers or instruction-tuned autoregressive language models trained on massive parallel multilingual corpora. These models handle diverse syntactic structures, idioms, and specialized technical terminology across hundreds of global languages. By understanding long-range discourse context, modern models prevent awkward phrasing. Translation models are critical for global e-commerce, international diplomacy, localized software distribution, subtitle generation, and breaking linguistic communication barriers worldwide.

## Representative Model: `CohereLabs/North-Small-Translate-1.0`

### Model profile
North-Small-Translate-1.0 is a 218.25B-parameter (25B active) sparse MoE translation model released in September 2026 under CC BY-NC 4.0. It is gated and non-commercial, covers 50 languages, and ships as BF16 (80 shards, ~436.5 GB) plus FP8 and NVFP4 W4A16 variants. It is a chat-style decoder, not a classic sequence-to-sequence model.

### Architecture tailored to translation
The decoder uses 128 experts with top-8 routing plus shared experts, a sigmoid router, and a 3:1 pattern of sliding-window (4,096) and global attention, with RoPE only in sliding layers. Context is 16K input and 16K output, and the model is explicitly non-reasoning. Sparse MoE keeps per-token compute near 25B while the total capacity covers 50 languages. See [mixture-of-experts](roadmap.md#mixture-of-experts).

### Niche mechanism
North uses a 128-expert top-8 sigmoid-routed MoE and alternates 4,096-token sliding attention with global attention; RoPE is applied only in the sliding layers. That keeps the 16K/16K translation context cheap, while the total 218B capacity carries 50 languages at ~25B active compute. **Design takeaway:** translation at scale is still an MoE + long-context attention problem; the data/RL recipe matters more than a novel block.

### Training and task tailoring
It builds on Command A+ in five stages: coarse SFT with four merged models (extending coverage from 35 to 50 languages), fine-grained synthetic MT SFT, DPO, online RL with GSPO and an LLM judge, then light targeted DPO at reduced learning rate to repair RL regressions. The WMT25 GEMBA-ESA progression was 71.3 → 76.2 → 80.1 → 80.0 → 81.8, showing the RL regression and recovery.

### How it is used
Use Transformers `AutoModelForCausalLM` with the chat template and greedy decoding, then parse replies with Cohere `melody` (`PyFilter`) to strip structural markers. vLLM needs `--tool-call-parser cohere_command4 --reasoning-parser cohere_command4`. BF16 needs 8×H100/4×B200, FP8 4×H100, and W4A16 2×H100.

### Fit and caveats
The tag fits, but this is a chat LM rather than a seq2seq translation pipeline. Reported WMT26 all-languages is 83.60 (84.36 agentic) versus Qwen3.5 397B-A17B at 81.56 and Google Translate at 68.20, but WMT26 is LLM-judged and unaudited. The non-commercial license is a hard constraint, and RL hurt some low-resource languages before the repair stage.

## Related Reading
- [Roadmap: mixture-of-experts](roadmap.md#mixture-of-experts) and [post-training](roadmap.md#post-training-sft-preferences-and-rl).
- [Text Generation](text-generation.md) · [Summarization](summarization.md) · [Token Classification](token-classification.md)
