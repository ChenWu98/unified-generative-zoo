# A Unified Interface for Guiding Generative Models (2D/3D GANs, Diffusion Models, and Their Variants)

Official PyTorch implementation of (**Section 4.3** of) our paper <br>
**Unifying Diffusion Models' Latent Space, with Applications to CycleDiffusion and Guidance** <br>
Chen Henry Wu, Fernando De la Torre <br>
Carnegie Mellon University <br>
_Preprint, Oct 2022_

[**[Paper link]**](https://arxiv.org/abs/2210.05559)

## Updates
**[Oct 13, 2022]** Code released.

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

## Contents

- [A Unified Interface for Guiding Generative Models (2D/3D GANs, Diffusion Models, and Their Variants)](#a-unified-interface-for-guiding-generative-models-2d3d-gans-diffusion-models-and-their-variants)
  - [Updates](#updates)
  - [TODOs](#todos)
  - [Notes](#notes)
  - [Overview](#overview)
  - [Contents](#contents)
  - [Dependencies](#dependencies)
  - [Pre-trained checkpoints](#pre-trained-checkpoints)
    - [Pre-trained generative models](#pre-trained-generative-models)
    - [Off-the-shelf models for guidance](#off-the-shelf-models-for-guidance)
  - [Usage](#usage)
    - [Overview](#overview-1)
    - [CLIP guidance for sampling sub-populations](#clip-guidance-for-sampling-sub-populations)
    - [Classifier guidance for sampling sub-populations](#classifier-guidance-for-sampling-sub-populations)
    - [ID guidance for sampling individuals](#id-guidance-for-sampling-individuals)
  - [Citation](#citation)
  - [License](#license)
  - [Contact](#contact)

## Dependencies

1. Create environment by running
```shell
conda env create -f environment.yml
conda activate generative_prompt
pip install git+https://github.com/openai/CLIP.git
```
2. Install `torch` and `torchvision` based on your CUDA version. 
3. Install [PyTorch 3D](https://github.com/facebookresearch/pytorch3d). Installing this library can be painful, but you can skip it if you are not using 3D GANs.
4. Install [taming-transformers](https://github.com/CompVis/taming-transformers) by running
```shell
cd ../
git clone git@github.com:CompVis/taming-transformers.git
cd taming-transformers/
pip install -e .
cd ../
```
5. Set up [wandb](https://wandb.ai/) for logging (registration is required). You should modify the ```setup_wandb``` function in ```main.py``` to accomodate your wandb credentials. You may want to run something like
```shell
wandb login
```

## Pre-trained checkpoints
### Pre-trained generative models
We provide a unified interface for various pre-trained generative models. Checkpoints for generative models used in this paper are provided below. 
1. StyleGAN2
```shell
cd ckpts/
wget https://www.dropbox.com/s/iy0dkqnkx7uh2aq/ffhq.pt
wget https://www.dropbox.com/s/lmjdijm8cfmu8h1/metfaces.pt
wget https://www.dropbox.com/s/z1vts069w683py5/afhqcat.pt
wget https://www.dropbox.com/s/a0hvdun57nvafab/stylegan2-church-config-f.pt
wget https://www.dropbox.com/s/x1d19u8zd6yegx9/stylegan2-car-config-f.pt
wget https://www.dropbox.com/s/hli2x42ekdaz2br/landscape.pt
```
2. StyleNeRF
```shell
cd ckpts/
wget https://www.dropbox.com/s/dtqsroh95uquwoc/StyleNeRF_ffhq_256.pkl
wget https://www.dropbox.com/s/klbuhqfv74q7e35/StyleNeRF_ffhq_512.pkl
wget https://www.dropbox.com/s/n80cr7isveh5yfu/StyleNeRF_ffhq_1024.pkl
```
3. Extended Analytic DPM
```shell
cd ckpts/
mkdir extended_adpm
cd extended_adpm/
wget https://www.dropbox.com/s/r8210seh6ekhogf/celeba64_ema_eps_epsc_pretrained_190000.ckpt.pth
wget https://www.dropbox.com/s/6o5etzhgbihr0yh/celeba64_ema_eps_eps2_pretrained_340000.ckpt.pth
wget https://www.dropbox.com/s/o0jw5ezai1e1z3v/celeba64_ema_eps.ckpt.pth
wget https://www.dropbox.com/s/0axtykkvyz49hrw/celeba64_ema_eps.ms_eps.pth
```
4. StyleGAN-XL
```shell
# StyleGAN-XL will be downloaded automatically. 
```
5. StyleSwin
```shell
cd ckpts/
wget https://www.dropbox.com/s/f0nlvu6fh3bbpmd/StyleSwin_FFHQ_1024.pt
wget https://www.dropbox.com/s/c2812gumbyxj751/StyleSwin_FFHQ_256.pt
```
6. StyleSDF
```shell
cd ckpts/
wget https://www.dropbox.com/s/epet782zdu0hazx/stylesdf_ffhq_vol_renderer.pt
wget https://www.dropbox.com/s/p0ptofh7sku2o8j/stylesdf_ffhq1024x1024.pt
wget https://www.dropbox.com/s/rq756clx14a9kgd/stylesdf_afhq_vol_renderer.pt
wget https://www.dropbox.com/s/hu5wgr40vyptzx6/stylesdf_afhq512x512.pt
wget https://www.dropbox.com/s/8rsaxzmey64jugo/stylesdf_sphere_init.pt
```
7. Diffusion Autoencoder
```shell
cd ckpts/
wget https://www.dropbox.com/s/ej0jj8g7crvtb5e/diffae_ffhq256.ckpt
wget https://www.dropbox.com/s/w5y89y57r9nd1jt/diffae_ffhq256_latent.pkl
wget https://www.dropbox.com/s/rsbpxaswnfzsyl1/diffae_ffhq128.ckpt
wget https://www.dropbox.com/s/v1dvsj6oklpz652/diffae_ffhq128_latent.pkl
```
8. Latent Diffusion Model
```shell
cd ckpts/
wget https://www.dropbox.com/s/9lpdgs83l7tjk6c/ldm_models.zip
unzip ldm_models.zip
```
9. NVAE
```shell
cd ckpts/
wget https://www.dropbox.com/s/bwwtszb5g5alw30/nvae_ffhq_256.pt
wget https://www.dropbox.com/s/8dfryaandkmoxzz/nvae_celebahq_256.pt
```
10. EG3D
```shell
cd ckpts/
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/zip -O eg3d.zip
unzip eg3d.zip
```
11. Denoising Diffusion GAN
```shell
cd ckpts/
wget https://www.dropbox.com/s/lsfkbln9u78rbhs/ddgan_celebahq256_netG_550.pth
```
12. GIRAFFE-HQ
```shell
cd ckpts/
wget https://www.dropbox.com/s/jj03hto6o9rnbha/giraffehd_ffhq_1024.pt
```
13. Diffusion-GAN
```shell
cd ckpts/
wget https://www.dropbox.com/s/25ryma8et4ohmjq/diffusion-stylegan2-ffhq.pkl
```

### Off-the-shelf models for guidance
1. CLIP
```text
# CLIP will be downloaded automatically
```
2. ArcFace IR-SE 50 model, provided by the Colab demo in [this repo](https://github.com/orpatashnik/StyleCLIP)
```shell
cd ckpts/
wget https://www.dropbox.com/s/qg7co4azsv5sacm/model_ir_se50.pth
```
3. CelebA classifier, trained by [this repo](https://github.com/ChenWu98/Generative-Visual-Prompt)
```shell
cd ckpts/
wget https://www.dropbox.com/s/yzc8ydaa4ggj1zs/celeba.zip
unzip celeba.zip 
```

## Usage

### Overview

Each set notation `{A,B,C}` stands for several independent experiments. 
You should always replace `{A,B,C}` with one of `A`, `B`, and `C`. 
Model checkpoints and image samples will be saved under `--output_dir`. 

### CLIP guidance for sampling sub-populations
1. ```Generative model``` $\in$ ```{LDM-DDIM, DiffAE, Diffusion-GAN, StyleGAN-XL, StyleGAN2, StyleNeRF, StyleSDF, EG3D, GIRAFFE-HD, StyleSwin, NVAE}```.
2. ```Dataset and resolution``` $\in$ ```{FFHQ1024, FFHQ512, FFHQ256, FFHQ128}```.
3. ```Text description``` $\in$ ```{"a photo of a baby", "a photo of an old person", "a photo of a person with eyeglasses", "a photo of a person with eyeglasses and a yellow hat"}```.
4. ```Guidance strength``` $\lambda_{\text{CLIP}}$ $\in$ ```{100, 300, 500, 700, 1000}```. In the following command, ```_500``` is omitted. 
5. Note that not all combinations of ```Generative model``` $\times$ ```Dataset and resolution``` are available. Please check the paper and [available configs](config/experiments) for details. 
```shell
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=clip_{a_baby,an_old_person,a_person_with_eyeglasses,a_person_with_eyeglasses_and_a_yellow_hat}_{ffhq1024,ffhq512,ffhq256,ffhq128}_{styleganxl,stylegan2,styleswin,stylenerf,latentdiff_5step,latentdiff_10step,diffae_3step_3step_latent_only,stylesdf,stylegan2_no_trunc,stylesdf_no_trunc,styleswin_no_trunc,styleganxl_no_trunc,stylenerf_no_trunc,nvae,eg3d,eg3d_no_trunc,giraffehd,diffae_3step_3step_both,diffusion_stylegan2,diffusion_stylegan2_no_trunc,diffae_10step_10step_both,}_langevin{,_100,_300,_700,_1000}
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1410 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &
```

### Classifier guidance for sampling sub-populations
```shell
# DDGAN CelebAHQ256 old
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=class_old_celebahq256_ddgan_langevin
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1430 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model ClassEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

# DDGAN CelebAHQ256 eyeglassess
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=class_eyeglasses_celebahq256_ddgan_langevin
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1430 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model ClassEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

# SN-DPM DDPM CelebA64 old
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=class_old_celeba64_sn_dpm_ddpm_langevin
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1430 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model ClassEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

# SN-DPM DDPM CelebA64 eyeglasses
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=class_eyeglasses_celeba64_sn_dpm_ddpm_langevin
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1430 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model ClassEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &
```


### ID guidance for sampling individuals
1. ```ID reference``` $\in$ ```{00001, 00002, 00015, 00018}```.
2. ```Generative model``` $\in$ ```{LDM-DDIM, DiffAE, StyleGAN-XL, StyleGAN2, DDGAN, EG3D, GIRAFFE-HD}```.
3. ```Dataset and resolution``` $\in$ ```{FFHQ1024, FFHQ512, FFHQ256, CelebAHQ256}```.
4. Note that not all combinations of ```Generative model``` $\times$ ```Dataset and resolution``` are available. Please check the paper and [available configs](config/experiments) for details. 

```shell
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=recon_id_{00001,00002,00015,00018}_{ffhq256,ffhq512,ffhq1024,celebahq256}_{latentdiff,diffae_10steps_10steps_both,giraffehd,stylegan2,styleganxl,ddgan,eg3d}_langevin
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1420 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 8 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &
```



## Citation
If you find this repository helpful, please cite as
```
@inproceedings{unifydiffusion2022,
  title={Unifying Diffusion Models' Latent Space, with Applications to {CycleDiffusion} and Guidance},
  author={Chen Henry Wu and Fernando De la Torre},
  booktitle={ArXiv},
  year={2022},
}
```
Do not forget to cite the original papers that proposed these models! 

## License
We use the X11 License. This license is identical to the MIT License, but with an extra sentence that prohibits using the copyright holders' names (Carnegie Mellon University in our case) for advertising or promotional purposes without written permission.




## Contact
[Issues](https://github.com/ChenWu98/unified-generative-zoo/issues) are welcome if you have any question about the code. 
If you would like to discuss the method, please contact [Chen Henry Wu](https://github.com/ChenWu98).

<a href="https://github.com/ChenWu98"><img src="https://avatars.githubusercontent.com/u/28187501?v=4"  width="50" /></a>
