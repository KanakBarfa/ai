# Keypoint Detection (`keypoint-detection`)

## Overview
Keypoint Detection identifies the precise two-dimensional or three-dimensional spatial coordinates of distinctive structural landmarks across objects, human bodies, or animal forms. Common targets include human skeletal joints, facial landmarks, hand knuckles, and vehicle chassis reference points. These models evaluate local visual patterns to output coordinate heatmaps or continuous point predictions. Keypoint detection provides the underlying tracking mechanism for human pose estimation, biomechanical motion analysis in athletics, interactive gaming without physical controllers, facial animation transfer in cinema production, and ergonomic safety monitoring in industrial manufacturing environments.

## Representative Model: `facebook/sapiens2-pose-1b`

### Model profile
Sapiens2-1B Pose is a 1.520B-parameter model (1.462B backbone), shipped as F32 safetensors (6.08 GB) in April 2026 under a custom Sapiens2 license. It predicts 308 whole-body keypoints—body, dense face, hands, and feet—for a human crop in Sociopticon format. It is not a COCO-17 pose model and does not include a person detector.

### Architecture tailored to whole-body pose
The backbone is a Sapiens2 ViT-1B: 40 layers, 1536 hidden, 24 heads (12 KV in middle layers), patch 16, 8 registers, QK-Norm, SwiGLU, and 4.7 TFLOPs. A top-down head uses two 4×4/stride-2 deconvolutions, 1×1 convolutions (768/512/256), and a 308-channel output with UDP stride 4 and sigma 6. High input resolution plus the dense head is what lets it resolve face and hand landmarks. See [Sapiens family](roadmap.md#model-family-notes).

### Niche mechanism
The top-down head is a heatmap decoder: two stride-2 deconvolutions raise 1/16 features to 1/4 resolution, 1×1 convolutions reduce channels, and 308 output channels encode body/face/hand/foot keypoints with UDP (stride 4, sigma 6). QK-norm and eight registers keep the 40-layer ViT stable at 1024×768. **Design takeaway:** for dense landmarks, use a high-res deconv head and a task-specific keypoint vocabulary; a detector crop is part of the system.

### Training and task tailoring
It shares the 1B-image Sapiens2 pretraining (MAE + contrastive). Pose fine-tuning uses 308 keypoints from capture-studio 3D data plus 100K in-the-wild annotations, with weighted MSE, OHEM, and a flip test. The demo runs an RTMDet detector first, then crops the person before pose inference.

### How it is used
Use the Sapiens2 repository (`scripts/demo/keypoints308.sh`) or HF `Sapiens2ForPoseEstimation`. The pipeline is detector → crop → 1024×768 pose inference. No quantization recipe is provided.

### Fit and caveats
The tag fits, but the model is human-only and top-down; it needs an external detector, uses a non-COCO 308-point vocabulary, and the card/paper disagree on face-keypoint counts (274 vs 243). Reported whole-body AP is 80.4 and AR 84.0 (foot 81.0, face 88.3, hands 77.6/78.0), versus Sapiens-1B 74.1; the 5B reaches 82.3 AP.

## Related Reading
- [Roadmap: Sapiens family](roadmap.md#model-family-notes) and [vision encoders](roadmap.md#vision-and-audio-encoders).
- [Depth Estimation](depth-estimation.md) · [Image Segmentation](image-segmentation.md) · [Robotics](robotics.md)
