# Unconditional Image Generation (`unconditional-image-generation`)

## Overview
Unconditional Image Generation refers to the synthesis of entirely novel, realistic images from pure random noise vectors without any conditioning inputs, guidance text, or reference imagery. The model learns the probability distribution of a specific visual dataset, such as human faces, landscapes, or medical radiographs, and generates new samples that closely mimic the statistical properties of the training data. While conditional models now dominate consumer creative tools, unconditional models remain essential for studying foundational generative modeling, synthetic dataset augmentation, privacy-preserving medical imaging research, and assessing the sample diversity of generative algorithms.

## Representative Model: `google/ddpm-cat-256`

### Model profile
This is the original DDPM cat checkpoint: a 113.67M-parameter U-Net in F32 (1.36 GB including samples), uploaded in July 2022 under Apache-2.0. It maps Gaussian noise of shape 3×256×256 to a 256×256 RGB cat image. There is no text, class, or image conditioning.

### Architecture tailored to unconditional synthesis
The `UNet2DModel` has six stages at 128/128/256/256/512/512 channels, two residual blocks per stage, attention in the second-to-last block (16×16), 32 normalization groups, SiLU activations, and a positional time embedding. The noise schedule is linear from 1e-4 to 0.02 over 1,000 timesteps, with `fixed_small` variance and `clip_sample=True`. It learns the LSUN Cat distribution with the DDPM simple noise-prediction objective and no fine-tuning, RL, or text alignment. See [diffusion foundations](roadmap.md#diffusion-flow-matching-and-vaes).

### Niche mechanism
DDPM's U-Net predicts the noise added at timestep `t`; the reverse process iteratively subtracts it over 1,000 steps. Time enters through a sinusoidal embedding added to each residual block; attention appears only at 16×16, so most compute is convolutional. `clip_sample` and `fixed_small` keep the reverse trajectory stable. **Design takeaway:** implement DDPM first—noise prediction + a U-Net + a linear schedule is the minimal diffusion system.

### How it is used
`DDPMPipeline.from_pretrained("google/ddpm-cat-256")()` samples noise and runs the full 1,000-step reverse process. `DDIMPipeline` or `PNDMPipeline` give faster sampling. It is FP32 and takes no prompt.

### Fit and caveats
The tag is accurate, but this is a 2022 cat-only 256×256 pixel-space model with slow sampling. The card's CIFAR-10 IS 9.46/FID 3.17 are DDPM-paper numbers, not results for this checkpoint; the guided-diffusion release reports LSUN Cat FID 5.57, precision 0.63, and recall 0.52. Diversity and resolution are low by 2026 standards, but it remains a clean teaching example.

## Related Reading
- [Roadmap: diffusion and flow matching](roadmap.md#diffusion-flow-matching-and-vaes).
- [Text-to-Image](text-to-image.md) · [Image-to-Image](image-to-image.md) · [Text-to-3D](text-to-3d.md)
