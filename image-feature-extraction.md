# Image Feature Extraction (`image-feature-extraction`)

## Overview
Image Feature Extraction converts raw input images into dense, low-dimensional continuous vector embeddings that summarize visual semantics, textures, and structural shapes. Trained through self-supervised objectives, contrastive learning, or masked autoencoding, these models produce representations where visually and semantically similar images map to neighboring coordinates in vector space. These continuous numerical representations act as foundational inputs for diverse downstream tasks, including high-scale reverse image search, duplicate image detection, clustering, visual recommendation engines, and serving as the visual front-end for multimodal foundation models.

## Representative Model: `facebook/dinov3-vit7b16-pretrain-lvd1689m`

### Model profile
DINOv3 ViT-7B/16 is a 6.716B-parameter vision backbone released in August 2025 under the gated DINOv3 License. It ships as F32 safetensors (26.9 GB). It maps an RGB image to a pooled CLS vector plus patch/register tokens; it is vision-only and not text-aligned, so it cannot do CLIP-style text-image retrieval.

### Architecture tailored to dense visual features
It is a ViT with embedding dimension 4096, 32 heads, patch size 16, RoPE, and four register tokens. A 224×224 image yields 1 CLS + 4 register + 196 patch tokens (201 total); larger images are accepted when both sides are multiples of 16. Registers and RoPE are what make the token features stable for dense tasks such as segmentation and depth, while the CLS token serves retrieval. See [vision encoders](roadmap.md#vision-and-audio-encoders).

### Niche mechanism
DINOv3 trains a student to match a momentum teacher on multi-crop views (DINO), adds iBOT masked-patch prediction, KoLeo on CLS, and Gram anchoring on patch features. Registers give high-norm tokens a dedicated sink, and RoPE makes patch features position-aware without learned position tables. The result is a frozen backbone whose patch tokens are useful for segmentation/depth, not just the CLS vector. **Design takeaway:** self-distillation + masked patches + Gram anchoring is a strong recipe for dense features; registers matter.

### Training and task tailoring
Pretraining is self-supervised on LVD-1689M (1.689B images from a 17B Instagram pool). The objectives combine DINO self-distillation with multi-crop, iBOT masked-image modeling, KoLeo regularization on CLS tokens, and Gram anchoring. Training used about 61,440 H100-hours (~18 t CO2eq). There is no SFT or RL.

### How it is used
Use `AutoImageProcessor` + `AutoModel`; `outputs.pooler_output` gives the CLS vector and `last_hidden_state` gives patch tokens. The weights are F32, so cast to BF16 for memory. Input sides must be multiples of 16.

### Fit and caveats
The tag fits, but the 7B/F32 checkpoint is heavy, gated, and custom-licensed. Reported results include IN-ReaL 90.4, ImageNet-R 91.1, ObjectNet 91.1, Oxford-H 72.8, ADE20k 55.9, NYU depth 0.309, DAVIS 79.7, and SPair 58.7. It is a feature extractor, not a classifier or captioner.

## Related Reading
- [Roadmap: vision encoders](roadmap.md#vision-and-audio-encoders) and [SAM3/DINOv3 family](roadmap.md#model-family-notes).
- [Image Classification](image-classification.md) · [Image Segmentation](image-segmentation.md) · [Video Classification](video-classification.md)
