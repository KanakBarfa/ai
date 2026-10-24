# Mask Generation (`mask-generation`)

## Overview
Mask Generation produces high-fidelity binary or categorical pixel masks for all identifiable structures within an image, often driven by promptable interactions such as points, bounding boxes, or rough scribbles. Rather than being restricted to a fixed list of object categories, promptable mask generators function in an open-world setting, delineating any requested subject or isolating every segment across an entire scene automatically. These architectures combine heavy image encoders with lightweight, high-speed mask decoders. Applications include interactive graphic design tools, scientific microscope cell segmentation, automated video rotoscoping, and robotic grasping target delineation.

## Representative Model: `facebook/sam3`

### Model profile
SAM 3 (Segment Anything Model 3) is an 848M-parameter promptable segmentation system released in November 2025 under a gated custom SAM 3 License. It works on images or video and accepts a concept prompt (noun phrase), an exemplar, or point/box/mask geometry, returning masks, boxes, scores, semantic masks, and track IDs. The Hub checkpoint is 859.9M F32 parameters in a 10.3 GB repository.

### Architecture tailored to promptable concepts
A shared Perception Encoder feeds a DETR-based detector that is conditioned jointly on text, geometry, and exemplars; a presence head decides whether the concept appears at all. A memory-based video tracker then propagates masks across frames. This combines open-vocabulary detection with class-agnostic mask decoding, so one model can find and segment a named concept instead of requiring a fixed label set. See [vision encoders](roadmap.md#vision-and-audio-encoders).

### Niche mechanism
SAM3 unifies detection and segmentation: a Perception Encoder feeds a DETR-style detector conditioned on text, boxes, points, or exemplars, and a presence head rejects absent concepts. A memory tracker propagates the chosen masks through video. The key is promptable concept queries, not class-agnostic "everything" masks. **Design takeaway:** use one promptable query interface for detection, segmentation, and tracking; train with hard negatives so the presence head is meaningful.

### Training and task tailoring
The model was trained with a data engine producing SA-Co: HQ contains 5.2M images and 4M noun phrases with hard negatives, SYN contains 38M phrases and 1.4B masks, and EXT adds 15 datasets. The catalogue covers 270K concepts (over 50× prior systems). No SFT or RL stages are stated.

### How it is used
For images, use `Sam3Model`/`Sam3Processor` with text or positive/negative boxes, then `post_process_instance_segmentation`. For video, use `Sam3VideoModel`; `sam3.pt` is also provided. Inference is FP32.

### Fit and caveats
The tag suggests "segment everything," but SAM3 is promptable concept segmentation; it needs a text, exemplar, or geometric prompt, and the tag hides the video/tracking capability. Reported SA-Co/Gold is 54.1 cgF1 versus 72.8 human, LVIS 48.5 AP, COCO box 56.4 AP, and SA-V 58.0 pHOTA. The license is gated and custom, and 860M FP32 is heavy.

## Related Reading
- [Roadmap: vision encoders](roadmap.md#vision-and-audio-encoders) and [SAM3/DINOv3 family](roadmap.md#model-family-notes).
- [Image Segmentation](image-segmentation.md) · [Zero-Shot Object Detection](zero-shot-object-detection.md) · [Keypoint Detection](keypoint-detection.md)
