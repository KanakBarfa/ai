# Token Classification (`token-classification`)

## Overview
Token Classification predicts a specific label for every individual token or word within an input text sequence. Rather than summarizing the entire document into one category, token classifiers assign tags based on local syntax and global context. The most prominent application is Named Entity Recognition, which extracts names of people, organizations, dates, and locations. Other key applications include part-of-speech tagging, clinical code extraction, and punctuation restoration. Token classification serves as an indispensable information extraction layer in legal intelligence, clinical record processing, and search query parsing.

## Representative Model: `openai/privacy-filter`

### Model profile
OpenAI Privacy Filter is a 1.40B-parameter token classifier (card: 1.5B total, 50M active) released in April 2026 under Apache-2.0. It labels text with 33 BIOES tags across a fixed taxonomy of eight PII categories plus background. BF16 and ONNX fp16/q4/q4f16 builds are provided in a ~17.4 GB repository.

### Architecture tailored to span labeling
It is a gpt-oss-style autoregressive checkpoint converted into a bidirectional token classifier: 8 pre-norm blocks, 640 hidden, 14 Q/2 KV grouped-query heads, 128 experts with top-4 routing, banded attention of 128 tokens (257-token effective window), and a 128K context. A classification head replaces the LM head, and constrained Viterbi decoding uses six transition biases. The banded bidirectional attention gives each token local left/right context while remaining efficient. See [masked vs. autoregressive objectives](roadmap.md#masked-vs-autoregressive-objectives).

### Niche mechanism
The base is an autoregressive gpt-oss MoE converted to a bidirectional token tagger: banded attention of 128 tokens (257 effective) lets each token see local left/right context, while 128 experts top-4 keep compute low. A token-level head replaces the LM head, and Viterbi decoding enforces valid BIOES transitions using six learned biases. **Design takeaway:** for PII spans, local bidirectional attention + constrained Viterbi is cheaper and more precise than a generative LLM.

### Training and task tailoring
It is autoregressively pretrained, then converted to bidirectional banded attention and trained with supervised token-level classification. The release includes `viterbi_calibration.json` for runtime precision/recall operating points. Fine-tuning is data-efficient: 10% of the training split reaches F1 0.962.

### How it is used
Use `pipeline("token-classification")`, `AutoModelForTokenClassification`, or Transformers.js with `dtype: "q4"` and WebGPU. Decode logits into spans with an aggregation strategy or the calibrated Viterbi decoder. It labels tokens; it does not emit redacted text.

### Fit and caveats
The tag fits. Reported PII-Masking-300k token P/R/F1 is 0.968/0.981/0.974 and span F1 0.942, versus a 0.940/0.980/0.960/0.926 baseline. CredData secret token F1 is 0.844 (span 0.624), and multilingual span F1 ranges from 0.758 (Hausa) to 0.934 (English). The taxonomy is fixed, English is primary, and performance drops on non-Latin scripts; treat it as a redaction aid, not a compliance guarantee.

## Related Reading
- [Roadmap: masked vs. autoregressive objectives](roadmap.md#masked-vs-autoregressive-objectives) and [post-training](roadmap.md#post-training-sft-preferences-and-rl).
- [Text Classification](text-classification.md) · [Question Answering](question-answering.md) · [Summarization](summarization.md)
