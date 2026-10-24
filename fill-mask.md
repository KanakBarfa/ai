# Fill-Mask (`fill-mask`)

## Overview
Fill-Mask models predict missing or occluded words within a sentence by evaluating bidirectional context from both left and right directions simultaneously. Championed by masked language modeling architectures like BERT and RoBERTa, this task serves as the foundational pretraining objective for deep representation learning. The model examines the surrounding semantic clues to determine the most grammatically sound and contextually appropriate token replacements. Beyond pretraining, fill-mask models are used in practical downstream applications such as automated grammatical error correction, assisted writing suggestions, masked token imputation in biological sequence analysis, and evaluating linguistic cloze test benchmarks.

## Representative Model: `facebook/esm2_t36_3B_UR50D`

### Model profile
ESM-2 3B is a 36-layer protein language model released in October 2022 under MIT. It ships as PyTorch `.bin` and TensorFlow `.h5` in FP32. It takes an amino-acid sequence and returns per-residue embeddings plus 33-way masked-token logits. The vocabulary is amino acids and special tokens, not natural-language subwords.

### Architecture tailored to protein sequences
It is a BERT-style bidirectional transformer: 36 layers, 2560 hidden, 40 heads, FFN 10,240, a 33-token vocabulary, rotary positions, and 1,026 max positions (1,024 residues plus specials), with an MLM head. Bidirectional context is essential because a residue's function depends on both N- and C-terminal neighbors. See [masked vs. autoregressive objectives](roadmap.md#masked-vs-autoregressive-objectives).

### Niche mechanism
ESM-2 is a bidirectional encoder with RoPE and an MLM head: it corrupts 15% of residues and predicts them from both N- and C-terminal context. Because the tokenizer is a 33-symbol amino-acid alphabet, one token is one residue, so attention has no subword fragmentation and the 1,024-position limit is a protein-length limit. **Design takeaway:** choose a tokenizer matched to the domain; for proteins, residue-level tokens + MLM is the whole recipe.

### Training and task tailoring
Pretraining is masked language modeling with 15% random masking on UniRef50/UR50D 2021_04. About 43M UniRef50 clusters sampled from ~138M UniRef90 sequences yield ~65M unique sequences over training. There is no SFT or RL. The MLM objective is the task itself, which is why the model transfers well to structure and function prediction.

### How it is used
Use `AutoTokenizer` + `AutoModelForMaskedLM`, or `pipeline("fill-mask")`, with `<mask>` inserted in the sequence. Input is capped at 1,024 residues, and logits are over 33 tokens. Encoder hidden states can be used as protein embeddings. It is FP32 only, with no official quantized release.

### Fit and caveats
`fill-mask` here means protein MLM, not NLP cloze. The vocabulary is 33 amino-acid tokens, input is capped at 1,024 residues, and the model is neither generative nor instruction-following. It is the 3B variant, below the 15B model. The family paper reports perplexity 10.45 (8M) vs 6.37 (15B) after 270k steps, and ESMFold folded 617M MGnify90 sequences with >225M high-confidence structures; there are no 3B-specific metrics on the card.

## Related Reading
- [Roadmap: masked vs. autoregressive objectives](roadmap.md#masked-vs-autoregressive-objectives) and [tokenization and embeddings](roadmap.md#tokenization-and-embeddings).
- [Feature Extraction](feature-extraction.md) · [Token Classification](token-classification.md) · [Text Generation](text-generation.md)
