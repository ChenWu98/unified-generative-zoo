# A Unified Interface for Guiding Generative Models <br> (2D/3D GANs, Diffusion Models, and Their Variants)

GANs, VAEs, and normalizing flows are usually characterized as deterministic mappings from **isometric Gaussian** latent codes to images. 
We show that it is possible to unify various diffusion models into this formulation. 
This allows us to guide (or condition, control) various generative models in a **unified, plug-and-play manner** by leveraging latent-space energy-based models (EBMs). 
This repository provides a unified interface for guiding various generative models with CLIP, classifiers, and face IDs. 
