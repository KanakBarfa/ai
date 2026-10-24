# Visual Question Answering (`visual-question-answering`)

## Overview
Visual Question Answering, commonly abbreviated as VQA, is a multimodal task where a system is provided an image alongside an open-ended natural language question concerning the visual content. The objective is to produce an accurate, context-aware answer. Achieving high performance in this task requires joint reasoning across visual semantics and textual logic, such as counting objects, interpreting relative spatial relationships, recognizing text embedded in scenes, and applying external commonsense knowledge. VQA models power accessibility tools for visually impaired individuals, automated medical image interpretation interfaces, customer support bots evaluating uploaded product photos, and visual intelligence inspection engines across autonomous robotics.

## Representative Model: `IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1`

### Model profile
Ziya-BLIP2-14B-Visual-v1 is a 2023 bilingual Chinese/English VQA model. The "14B" is a 13B Ziya-LLaMA-13B language model plus a BLIP-2 vision stack; the Hub repository holds only the vision-side weights (about 8.76 GB FP32), so the base LLM must be downloaded separately. It is released under GPL-3.0.

### Architecture tailored to VQA
A frozen ViT-g/14 extracts image features, a frozen 12-layer Q-Former with 32 learned queries compresses them, and a linear projection maps the result into the frozen Ziya-LLaMA-13B (40 layers, 5120 hidden, 2048-token context). Only the projection is trained. The Q-Former's learned queries are the key task adaptation: they select a small set of visually salient features for the language model instead of forcing it to attend to every patch. Multi-image and multi-turn chat are supported.

### Niche mechanism
BLIP-2 uses a frozen ViT-g/14, a frozen 12-layer Q-Former with 32 learned queries, and a linear projection into a frozen 13B LLaMA. Only the projection trains, so vision-language alignment is learned in a very small parameter budget; the Q-Former compresses hundreds of patches into 32 tokens. **Design takeaway:** a learned query bottleneck + frozen towers is a cheap connector; it is also why this 2023 model is obsolete.

### Training and task tailoring
Training is data-centric and two-stage, with vision encoder and LLM frozen. Stage one aligns image features to the LLM using captions; stage two instruction-tunes on VQA and dialogue. The corpus is about 20M cleaned Chinese/English samples assembled from open VQA sets, translated bilingual data, and BLIP/Grounded-SAM-generated captions. No RLHF stage is reported.

### How it is used
Usage is custom rather than a standard pipeline: load `Ziya-LLaMA-13B-v1` as `language_model` inside `ZiyaBlip2ForCausalLM`, then call `model.chat()`. Typical settings are temperature 0.7, top_p 0.1, and 128 new tokens, with old torch/tokenizers versions and FP32.

### Fit and caveats
The VQA tag fits, but the release is incomplete without the LLM, the code is obsolete, context is only 2K, and only Chinese/English are supported. GPL-3.0 is copyleft. For new work, a modern open VLM is a better default.

## Related Reading
- [Roadmap: multimodal fusion](roadmap.md#multimodal-fusion) and [vision encoders](roadmap.md#vision-and-audio-encoders).
- [Image-to-Text](image-to-text.md) · [Image-Text-to-Text](image-text-to-text.md) · [Document Question Answering](document-question-answering.md)
