# Image-to-3D (`image-to-3d`)

## Overview
Image-to-3D models reconstruct detailed three-dimensional object geometry and surface textures from one or more two-dimensional reference images. The architecture infers hidden viewpoints, occluded surfaces, and realistic depth profiles to synthesize complete polygon meshes or 3D Gaussian representations. By eliminating the manual multi-day modeling process traditionally required in computer graphics software, Image-to-3D pipelines enable rapid 3D asset generation from single product photos or concept art. Primary applications include e-commerce product visualization, game asset development, virtual production, augmented reality spatial experiences, and rapid prototyping.

## Representative Model: `microsoft/TRELLIS.2-4B`

### Model profile
TRELLIS.2-4B is a roughly 4B-parameter image-to-3D system released in December 2025 under MIT. It combines flow-matching diffusion transformers with a sparse 3D VAE and produces a textured mesh with PBR materials from a single RGB or reference image. The repository is about 16.2 GB and uses the custom `trellis2` library.

### Architecture tailored to 3D asset generation
It uses a "field-free" O-Voxel representation that encodes geometry, normals, and PBR properties, with a Sparse Compression VAE at 16× spatial downsampling. A 1024³ asset becomes about 9.6K latent tokens, and flow-matching DiTs (about 1.3B each, ~4B total) generate the structured latents. This keeps 3D generation tractable at 512³–1536³ voxels while preserving material detail. See [diffusion and flow matching](roadmap.md#diffusion-flow-matching-and-vaes).

### Niche mechanism
O-Voxel stores geometry, normals, and PBR on a sparse voxel grid, so the model never fits a dense neural field. A sparse compression VAE downsamples 16× and packs a 1024³ asset into about 9.6K latent tokens; flow-matching DiTs generate those tokens, and the decoder returns a textured mesh. **Design takeaway:** make the 3D representation sparse and latent before diffusion; dense radiance fields waste compute on empty space.

### Training and task tailoring
The SC-VAE is trained on filtered Trellis-500K assets retaining PBR, and the generative model is trained on about 800K assets augmented with TexVerse. The DiTs used 32 H100s at learning rate 1e-4. There is no RLHF or preference alignment.

### How it is used
Use `Trellis2ImageTo3DPipeline.from_pretrained(...)` with a single image, then export via `o_voxel.postprocess.to_glb` with 4096 textures. It requires Linux, CUDA 12.4, and at least 24 GB VRAM. Generation takes about 3/17/60 s at 512/1024/1536³.

### Fit and caveats
The tag fits. Reported scores are CLIP 0.894 (vs 0.876), ULIP-2 0.477 (vs 0.474), Uni3D 0.436 (vs 0.427), and a 66.5% vs 13.3% user preference; PBR reconstruction reaches 38.89 dB PSNR / 0.033 LPIPS. Raw meshes can contain small holes, and there is no preference alignment, so production assets need cleanup.

## Related Reading
- [Roadmap: diffusion and flow matching](roadmap.md#diffusion-flow-matching-and-vaes) and [transformers](roadmap.md#transformers-and-attention).
- [Text-to-3D](text-to-3d.md) · [Depth Estimation](depth-estimation.md) · [Image-to-Image](image-to-image.md)
