# Text-to-Audio (`text-to-audio`)

## Overview
Text-to-Audio models synthesize rich non-speech audio such as music, Foley sound effects, and ambient environmental soundscapes from written descriptive prompts. These models use latent audio diffusion or autoregressive sequence generation over discrete residual vector quantized audio codecs. Content creators can describe precise instruments, tempo, and ambient atmospheres to generate custom audio programmatically without recording equipment. The architecture ensures temporal coherence and frequency balance across complex audio textures. Key applications include video game dynamic sound design, film post-production background score synthesis, sound effect generation for multimedia, and virtual reality acoustic simulation.

## Representative Model: `ACE-Step/acestep-captioner`

### Model profile
ACE-Step Captioner is a 10.73B-parameter Qwen2.5-Omni derivative (10.28B BF16 plus 449M F32, about 22.4 GB) released in early 2026 under MIT. Despite the `text-to-audio` tag, it runs **audio → text**: it listens to music or sound and writes a structured description. It is a data-labeling component for the ACE-Step music-generation family, not a sound generator.

### Architecture and task tailoring
The model uses `Qwen2_5OmniForConditionalGeneration`: a 7B-class thinker (28 layers, 3584 hidden, 32-layer/1280-wide audio encoder) plus a 24-layer talker that is present in the config but unused for captioning. The audio encoder and thinker are the relevant path; the talker would only matter for speech output. See [multimodal fusion](roadmap.md#multimodal-fusion).

### Niche mechanism
The tag direction is reversed: the 32-layer/1280 audio encoder turns audio into features for the Qwen2.5-Omni thinker, which generates a structured caption; the 24-layer talker is unused. Training uses SFT on 5M Gemini annotations, reward models on 4M synthetic negative pairs, and RL refinement. **Design takeaway:** a multimodal LLM can label audio at scale, but generation needs a separate codec/diffusion decoder.

### Training and task tailoring
Per arXiv 2602.00744, the model was SFT-trained from Qwen2.5-Omni on a 5M-sample Gemini-2.5-Pro-annotated Golden Set, then reward models were trained on 4M synthetic negative pairs and used for RL refinement; 27M samples were annotated. A local GRPO run used lr 1e-5, 3 epochs/10k steps, and BF16. The output schema covers style, instruments, structure, and timbre.

### How it is used
Use it like Qwen2.5-Omni-7B via `AutoProcessor` / `AutoModelForMultimodalLM`: pass audio plus a prompt such as `*Task* Describe this audio in detail` and `<audio>`, then generate text. BF16 is recommended; no special serving or quantization is documented.

### Fit and caveats
The tag points the wrong way; this cannot generate audio. The card claims better music description than Gemini Pro 2.5, with 1000+ instruments, 1000+ descriptive terms, and structure/timbre analysis, but gives no numeric table or protocol. It is useful for auto-labeling music datasets, not for text-to-audio synthesis.

## Related Reading
- [Roadmap: task-tag caveats](roadmap.md#task-tag-caveats) and [multimodal fusion](roadmap.md#multimodal-fusion).
- [Text-to-Speech](text-to-speech.md) · [Audio-Text-to-Text](audio-text-to-text.md) · [Audio Classification](audio-classification.md)
