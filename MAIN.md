# Hugging Face Model Tasks

This document catalogues all 52 model task types available on the Hugging Face Hub, grouped by domain. Each entry provides the official pipeline tag and a concise two-sentence explanation of what the task entails.

**Start here:** [roadmap.md](roadmap.md) explains the shared concepts, exact mechanisms, model families, and a dependency-based reading order. Each task file is a compact spec: representative model, exact architecture, the niche mechanism that makes it work, training, use, and caveats.

---

## Multimodal

### 1. `any-to-any` (Any-to-Any)
Any-to-Any models process arbitrary combinations of modalities such as text, audio, images, and video as inputs. They generate outputs across one or more target modalities based on unified cross-modal representations.

### 2. `visual-question-answering` (Visual Question Answering)
Visual Question Answering models receive an image alongside a natural language question about its visual content. The model analyzes visual elements to generate an accurate textual answer.

### 3. `document-question-answering` (Document Question Answering)
Document Question Answering models interpret visually rich documents such as scanned PDFs, invoices, and forms. They combine text extraction and layout understanding to answer specific questions regarding the document contents.

### 4. `visual-document-retrieval` (Visual Document Retrieval)
Visual Document Retrieval models search and rank visually rich documents based on text queries or visual prompts. They analyze layout, typography, and embedded graphics to find relevant document pages.

### 5. `audio-text-to-text` (Audio-Text-to-Text)
Audio-Text-to-Text models take combined spoken audio and text prompts as multimodal input. They produce textual responses to accomplish conversational speech understanding, reasoning, and multimodal question answering.

### 6. `image-text-to-text` (Image-Text-to-Text)
Image-Text-to-Text models accept image inputs along with textual queries to perform visual reasoning. They output detailed textual descriptions, explanations, or answers conditioned on both modalities.

### 7. `image-text-to-image` (Image-Text-to-Image)
Image-Text-to-Image models modify or synthesize images guided by both a reference image and descriptive text instructions. They enable controllable visual editing, style alterations, and subject-driven image composition.

### 8. `image-text-to-video` (Image-Text-to-Video)
Image-Text-to-Video models generate video sequences starting from a static image and a descriptive text prompt. They animate the source imagery while following motion dynamics and scenes outlined in the text.

### 9. `video-text-to-text` (Video-Text-to-Text)
Video-Text-to-Text models process video sequences together with user text prompts to analyze temporal visual data. They return text responses summarizing events, tracking actions, or answering questions across video frames.

---

## Computer Vision

### 10. `image-classification` (Image Classification)
Image Classification models assign one or more predefined category labels to an entire input image. They identify the primary subject or scene depicted without locating specific object boundaries.

### 11. `image-segmentation` (Image Segmentation)
Image Segmentation models partition an image into distinct visual regions by assigning semantic labels to individual pixels. This task includes semantic, instance, and panoptic segmentation to delineate object shapes precisely.

### 12. `zero-shot-image-classification` (Zero-Shot Image Classification)
Zero-Shot Image Classification models classify images into novel categories without requiring explicit task-specific training data. They match image embeddings with natural language class descriptions using shared representation spaces.

### 13. `object-detection` (Object Detection)
Object Detection models identify the presence and spatial locations of objects within an image. The model outputs rectangular bounding boxes and category labels with confidence scores for each detected item.

### 14. `zero-shot-object-detection` (Zero-Shot Object Detection)
Zero-Shot Object Detection models locate and classify objects described through free-form text labels without prior category training. They ground user-specified textual queries directly to bounding boxes within the visual scene.

### 15. `depth-estimation` (Depth Estimation)
Depth Estimation models predict the relative or metric distance of every scene point from the camera viewpoint. They output a grayscale depth map derived from single monocular images or stereo pairs.

### 16. `keypoint-detection` (Keypoint Detection)
Keypoint Detection models identify specific anatomical joints, facial landmarks, or distinct structural interest points in images. They provide coordinates useful for human pose estimation, motion tracking, and gesture recognition.

### 17. `mask-generation` (Mask Generation)
Mask Generation models produce binary or categorical pixel masks for all identifiable objects or designated regions in an image. They support open-world promptable segmentation guided by spatial points, bounding boxes, or automated grid searches.

### 18. `image-to-image` (Image-to-Image)
Image-to-Image models transform an existing source image into a modified version while retaining core structural content. Common applications include super-resolution, inpainting, style transfer, colorization, and image restoration.

### 19. `unconditional-image-generation` (Unconditional Image Generation)
Unconditional Image Generation models synthesize completely new images from random noise vectors without condition inputs or text prompts. They learn the underlying distribution of a training dataset to produce realistic samples of a target domain.

### 20. `text-to-image` (Text-to-Image)
Text-to-Image models generate photorealistic or artistic images directly from natural language descriptive prompts. They use diffusion or autoregressive techniques to align visual composition with fine-grained semantic details.

### 21. `image-to-text` (Image-to-Text)
Image-to-Text models inspect an input image and generate corresponding descriptive language or transcribed text. Typical use cases include automated image captioning and optical character recognition for reading text in scenes.

### 22. `image-feature-extraction` (Image Feature Extraction)
Image Feature Extraction models map input images into dense low-dimensional numerical embeddings. These latent vector representations serve as inputs for downstream tasks like visual search, clustering, and nearest-neighbor retrieval.

### 23. `text-to-3d` (Text-to-3D)
Text-to-3D models convert natural language prompts into three-dimensional digital assets such as meshes or neural radiance fields. They produce textured 3D geometric models suitable for animation, gaming, and virtual reality workflows.

### 24. `image-to-3d` (Image-to-3D)
Image-to-3D models reconstruct full 3D geometry and surface textures from single or sparse 2D image inputs. They infer unseen angles and depth to export complete polygon meshes or 3D Gaussian splats.

### 25. `video-classification` (Video Classification)
Video Classification models analyze sequences of video frames to assign high-level category labels to the entire clip. They capture spatial features and temporal dynamics to recognize sports actions, activities, and scene contexts.

### 26. `text-to-video` (Text-to-Video)
Text-to-Video models synthesize multi-frame animated video clips from descriptive textual prompts. They generate coherent motion trajectories and consistent visual scenes across consecutive video frames.

### 27. `image-to-video` (Image-to-Video)
Image-to-Video models take a static source image and generate plausible motion frames extending from it. They animate characters, camera pans, and natural physics while maintaining the original visual identity.

### 28. `video-to-video` (Video-to-Video)
Video-to-Video models transform an existing video stream into a new visual style while preserving motion and structure. They are frequently used for video stylization, temporal super-resolution, and sequential inpainting.

---

## Natural Language Processing

### 29. `text-classification` (Text Classification)
Text Classification models assign predefined categories to full text documents, paragraphs, or sentences. Common applications include sentiment analysis, spam detection, topic tagging, and intent classification.

### 30. `token-classification` (Token Classification)
Token Classification models predict a specific label for each individual token or word within a sequence. This task underpins named entity recognition, part-of-speech tagging, and punctuation restoration.

### 31. `question-answering` (Question Answering)
Question Answering models extract or synthesize answers to natural language questions based on provided reference text or external knowledge. They support both extractive formats that highlight exact spans and abstractive formats that generate free-form responses.

### 32. `table-question-answering` (Table Question Answering)
Table Question Answering models query structured tabular data in response to natural language inquiries. They perform reasoning, aggregation, and cell retrieval across columns and rows to return direct answers.

### 33. `zero-shot-classification` (Zero-Shot Classification)
Zero-Shot Classification models categorize text sequences into candidate classes that were never seen during model training. They formulate classification as a natural language inference problem by testing premise and hypothesis entailment.

### 34. `translation` (Translation)
Translation models convert written text from one human language into another while preserving semantic meaning. They handle both bilingual pairs and multilingual settings across diverse syntax rules and vocabularies.

### 35. `summarization` (Summarization)
Summarization models condense lengthy documents into concise overviews containing only the most critical information. They generate either extractive summaries using key source sentences or abstractive summaries using newly synthesized phrasing.

### 36. `feature-extraction` (Feature Extraction)
Feature Extraction models encode input text into high-dimensional vector representations capturing semantic nuances. The generated embeddings are utilized for semantic search, vector database indexing, and downstream machine learning models.

### 37. `text-generation` (Text Generation)
Text Generation models autoregressively predict subsequent tokens to produce coherent, contextually relevant text passages. They power conversational assistants, code completion tools, creative writing systems, and reasoning engines.

### 38. `fill-mask` (Fill-Mask)
Fill-Mask models predict missing words masked within a sentence by evaluating bidirectional context. They serve as foundational pretraining tasks for masked language models and assist in lexical error correction.

### 39. `sentence-similarity` (Sentence Similarity)
Sentence Similarity models determine how closely two text passages align in meaning and intent. They produce numerical similarity scores by comparing sentence embeddings using cosine similarity or distance metrics.

### 40. `text-ranking` (Text Ranking)
Text Ranking models evaluate a list of candidate documents and reorder them according to relevance for a given search query. They are essential components in multi-stage retrieval pipelines and search engine rerankers.

---

## Audio

### 41. `audio-classification` (Audio Classification)
Audio Classification models assign categorical labels to raw audio clips or spectrogram representations. They identify acoustic scenes, musical genres, speaker identities, and environmental sounds.

### 42. `automatic-speech-recognition` (Automatic Speech Recognition)
Automatic Speech Recognition models transcribe spoken voice recordings into written text across various languages. They power voice search, automated captioning, and real-time meeting transcription systems.

### 43. `text-to-speech` (Text-to-Speech)
Text-to-Speech models synthesize natural-sounding spoken audio waveforms from written textual input. They offer customizable voice profiles, emotional tones, and pronunciation controls for voice assistants and audiobooks.

### 44. `text-to-audio` (Text-to-Audio)
Text-to-Audio models generate diverse non-speech audio such as music, sound effects, and ambient noise from text prompts. They enable content creators to synthesize custom soundscapes and instrumental compositions programmatically.

### 45. `audio-to-audio` (Audio-to-Audio)
Audio-to-Audio models map an input audio track into another modified audio waveform. They handle tasks such as speech enhancement, source separation, voice conversion, and background noise removal.

### 46. `voice-activity-detection` (Voice Activity Detection)
Voice Activity Detection models detect the presence or absence of human speech within continuous audio streams. They filter out silence and background noise to optimize upstream speech recognition and communication bandwidth.

---

## Tabular

### 47. `tabular-classification` (Tabular Classification)
Tabular Classification models predict discrete target labels based on structured rows containing numerical and categorical features. Typical applications include credit risk assessment, churn prediction, and medical diagnostic classification.

### 48. `tabular-regression` (Tabular Regression)
Tabular Regression models predict continuous numerical outcomes from structured tabular datasets. They are used for financial forecasting, house price valuation, and demand projection.

---

## Reinforcement Learning

### 49. `reinforcement-learning` (Reinforcement Learning)
Reinforcement Learning models train decision-making agents to select optimal actions within dynamic environments to maximize cumulative rewards. They are applied in game playing, autonomous navigation, algorithmic trading, and system optimization.

---

## Robotics

### 50. `robotics` (Robotics)
Robotics models translate sensory perceptions and instructions into physical control signals and trajectory commands for robotic systems. They enable robotic arms, mobile platforms, and manipulators to grasp objects and navigate complex real-world environments.

---

## Graph Machine Learning

### 51. `graph-ml` (Graph Machine Learning)
Graph Machine Learning models process relational graph structures composed of interconnected nodes and edges. They perform node classification, link prediction, and molecular property prediction for social networks and bioinformatics.

---

## Time Series

### 52. `time-series-forecasting` (Time Series Forecasting)
Time Series Forecasting models predict future values along sequential data points recorded over regular time intervals. They help organizations anticipate energy consumption, stock trends, weather patterns, and supply chain inventory needs.
