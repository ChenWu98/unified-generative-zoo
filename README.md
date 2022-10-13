# A Unified Interface for Guiding Generative Models <br> (2D/3D GANs, Diffusion Models, and Their Variants)

Official PyTorch implementation of (**Section 4.3** of) our paper <br>
**Unifying Diffusion Models' Latent Space, with Applications to CycleDiffusion and Guidance** <br>
Chen Henry Wu, Fernando De la Torre <br>
Carnegie Mellon University <br>
_Preprint, Oct 2022_

[**[Paper link]**](https://arxiv.org/abs/2210.05559)

## Notes
1. **Sections 4.1** and **4.2** of this paper is open-sourced at [CycleDiffusion](https://github.com/ChenWu98/cycle-diffusion).
2. The code is based on [Generative Visual Prompt](https://github.com/ChenWu98/Generative-Visual-Prompt).
3. Feel free to email me if you think I should cite your work! 

## Overview

GANs, VAEs, and normalizing flows are usually characterized as deterministic mappings from **isometric Gaussian** latent codes to images. 
We show that it is possible to unify various diffusion models into this formulation. 
This allows us to guide (or condition, control) various generative models in a **unified, plug-and-play manner** by leveraging latent-space energy-based models (EBMs). 
This repository provides a unified interface for guiding various generative models with CLIP, classifiers, and face IDs. 

Models studied in this paper (some of them are not included here; please check [CycleDiffusion](https://github.com/ChenWu98/cycle-diffusion)):

<div align=center>
    <img src="docs/models.png" align="middle" width=750>
</div>

<br>

An illustration of generative models as deterministic mappings from isometric Gaussian latent codes to images. 

<div align=center>
    <img src="docs/convert.png" align="middle" width=750>
</div>

<br>

Interestingly, we find that different models represent subpopulations and individuals in different ways, although most of them are trained on the same data. 

<div align=center>
    <img src="docs/samples.png" align="middle" width=750>
</div>

<br>

<div align=center>
    <img src="docs/ids.png" align="middle" width=725>
</div>