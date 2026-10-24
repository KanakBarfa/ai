# Depth Estimation (`depth-estimation`)

## Overview
Depth Estimation predicts the spatial distance between the camera lens and every physical point depicted in a scene, converting two-dimensional RGB imagery into three-dimensional geometric depth maps. Models accomplish this either monocularly from a single viewpoint by inferring perspective cues, shadows, and object scale, or binocularly using stereo disparity matching. High-quality depth prediction bridges the gap between planar photographs and spatial computing environments. It is essential for autonomous vehicle navigation, drone collision avoidance, augmented reality occlusion rendering, synthetic depth-of-field portrait photography, and three-dimensional reconstruction pipelines across architectural scanning.

## Representative Model: `facebook/sapiens2-pointmap-5b`

### Model profile
Sapiens2-5B Pointmap is a 5.348B-parameter model (5.071B backbone), shipped as F32 safetensors (21.4 GB) in April 2026 under a custom Sapiens2 license. It maps a 1024×768 RGB image to a per-pixel 3D pointmap `(x, y, z)` in the camera frame; it does not output a conventional grayscale depth map.

### Architecture tailored to 3D pointmaps
The backbone is a Sapiens2 ViT-5B: 56 layers, 2432 hidden, 32 heads (16 KV in middle GQA layers), patch 16, 8 registers, QK-Norm, SwiGLU, and 15.7 TFLOPs. The head uses 3×3 convolutions (128/64/32), scale convolutions, pixel-shuffle, four upsampling stages, and three output channels. It predicts focal-normalized XYZ plus a scalar scale, which is the key adaptation for metric-ambiguous single-image geometry. See [Sapiens family](roadmap.md#model-family-notes).

### Niche mechanism
The pointmap head does not regress grayscale depth: it upsamples with pixel-shuffle to 1024×768 and emits XYZ plus a scalar scale, so the network represents camera-frame geometry while leaving absolute scale ambiguous. QK-norm and eight register tokens stabilize the 56-layer ViT; training uses synthetic labels with L2 on XYZ and an XY-gradient term to preserve edges. **Design takeaway:** predict 3D points directly and factor scale separately; dense geometry needs high-resolution features, not a global pool.

### Training and task tailoring
The backbone is pretrained on 1B human images with a joint MAE (75% masking) and contrastive objective (MAE 1.0, CLS 0.4, KoLeo 0.04). Pointmap post-training uses about 1M synthetic labels with L2 loss on XYZ plus an XY gradient loss; no real depth labels are used, so the model learns geometry from rendered/synthetic supervision.

### How it is used
Use the Sapiens2 repository (`scripts/demo/pointmap.sh`) or the HF `Sapiens2ForPointmapEstimation` class with `AutoImageProcessor` at 1024×768. Expect FP32 memory use; no quantized recipe is published.

### Fit and caveats
The tag is depth-estimation, but the output is a pointmap with ambiguous scale, and the model is human-centric. Supervision is synthetic-only, F32 weights are 21 GB, and the custom license restricts surveillance and biometric use. Reported test distance is 0.167 and absolute error 0.443, versus MoGe 0.202/0.486.

## Related Reading
- [Roadmap: Sapiens family](roadmap.md#model-family-notes) and [vision encoders](roadmap.md#vision-and-audio-encoders).
- [Keypoint Detection](keypoint-detection.md) · [Image-to-3D](image-to-3d.md) · [Image Segmentation](image-segmentation.md)
