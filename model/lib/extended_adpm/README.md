# Extended Analytic-DPM

* This is the official implementation for [Estimating the Optimal Covariance with Imperfect Mean in Diffusion Probabilistic Models](https://arxiv.org/abs/2206.07309) (Accepted in ICML 2022). It extends [Analytic-DPM](https://arxiv.org/abs/2201.06503) under the following two settings:
    * The reverse process adpots complicated covariance matrices dependent to states, instead of simple scalar variances (which motivates the SN-DPM in the paper).
    * The score-based model has some error w.r.t. the exact score function (which motivates NPR-DPM in the paper).

* This codebase also reimplements [Analytic-DPM](https://arxiv.org/abs/2201.06503) and reproduces its most results. The pretrained DPMs used in the Analytic-DPM paper are provided <a href="#pretrained_dpm">here</a>, and have already been converted to a format that can be directly used for this codebase. We also additionally applies Analytic-DPM to score-based SDE.

* Models and FID statistics are available <a href="#model">here</a> to reproduce results in this paper.


## Dependencies
The codebase is based on `pytorch`. The dependencies are listed below.
```sh
pip install pytorch>=1.9.0 torchvision ml-collections ninja tensorboard
```


## Basic usage

The basic usage for training is
```sh
python run_train.py --pretrained_path path/to/pretrained_dpm --dataset dataset --workspace path/to/working_directory $train_hparams
```
* `pretrained_path` is the path to a pretrained diffusion probabilistic model (DPM). <a href="#pretrained_dpm">Here</a> provide all pretrained DPMs used in this work.
* `dataset` represents the training dataset, one of <`cifar10`|`celeba64`|`imagenet64`|`lsun_bedroom`>.
* `workspace` is the place to put training outputs, e.g., logs and middle checkpoints.
* `train_hparams` specify other hyperparameters used in training. <a href="#model">Here</a> lists `train_hparams` for all models.


The basic usage for evaluation is
```sh
python run_eval.py --pretrained_path path/to/evaluated_model --dataset dataset --workspace path/to/working_directory \
    --phase phase --sample_steps sample_steps --batch_size batch_size --method method $eval_hparams
```
* `pretrained_path` is the path to a model to evaluate. <a href="#model">Here</a> provide all models evaluated in this work.
* `dataset` represents the dataset the model is trained on, one of <`cifar10`|`celeba64`|`imagenet64`|`lsun_bedroom`>.
* `workspace` is the place to put evaluation outputs, e.g., logs, samples and bpd values.
* `phase` specifies running sampling or likelihood evaluation, one of <`sample4test`|`nll4test`>.
* `sample_steps` is the number of steps to run during inference, the samller this value the faster the inference.
* `batch_size` is the batch size, e.g., 500.
* `method` specifies the type of the model, one of:
    * `pred_eps` the original DPM (i.e., a noise prediction model) with discrete timesteps
    * `pred_eps_eps2_pretrained` the SN-DPM with discrete timesteps
    * `pred_eps_epsc_pretrained` the NPR-DPM with discrete timesteps
    * `pred_eps_ct2dt` the original (i.e., a noise prediction model) with continuous timesteps (i.e., a score-based SDE)
    * `pred_eps_eps2_pretrained_ct2dt` the SN-DPM with continuous timesteps
    * `pred_eps_epsc_pretrained_ct2dt` the NPR-DPM with continuous timesteps
* `eval_hparams` specifies other optional hyperparameters used in evaluation.
* <a href="#evaluation">Here</a> lists `method` and `eval_hparams` for NPR/SN-DPM and Analytic-DPM results in this paper.



## Models and FID statistics
<a id="model"/>

Here is the list of NPR-DPMs and SN-DPMs trained in this work. These models only train an additional prediction head in the last layer of a pretrained diffusion probabilistic model (DPM).

| NPR/SN-DPM | Pretrained DPM | `train_hparams` |
|:----:|:----:|:----:|
|[CIFAR10 (LS), NPR-DPM](https://drive.google.com/file/d/1qP1kcjz6fSzevWSEWFucJS7gDWk88-35/view?usp=sharing) | [CIFAR10 (LS)](https://drive.google.com/file/d/1rhZBWUDK3_q37Iac3sXq6WnxR_OHhPyI/view?usp=sharing) | `"--method pred_eps_epsc_pretrained"` |
|[CIFAR10 (LS), SN-DPM](https://drive.google.com/file/d/1lyKwqilnIvYIBT2M7THAAQMTqMJ8-paX/view?usp=sharing) | [CIFAR10 (LS)](https://drive.google.com/file/d/1rhZBWUDK3_q37Iac3sXq6WnxR_OHhPyI/view?usp=sharing) | `"--method pred_eps_eps2_pretrained"` |
|[CIFAR10 (CS), NPR-DPM](https://drive.google.com/file/d/1Hf4keE-akBcygQIdhEBGnSYOxJPltPYj/view?usp=sharing) | [CIFAR10 (CS)](https://drive.google.com/file/d/1ONNLpqPDLr4NesC0TfVZ3dCyaVBu7Xw0/view?usp=sharing) | `"--method pred_eps_epsc_pretrained --schedule cosine_1000"` |
|[CIFAR10 (CS), SN-DPM](https://drive.google.com/file/d/1DeQ2EWBwV91akURFYgcZ3ObLr0gvCnIZ/view?usp=sharing) | [CIFAR10 (CS)](https://drive.google.com/file/d/1ONNLpqPDLr4NesC0TfVZ3dCyaVBu7Xw0/view?usp=sharing) | `"--method pred_eps_eps2_pretrained --schedule cosine_1000"` |
|[CIFAR10 (VP SDE), NPR-DPM](https://drive.google.com/file/d/1wcZ0U3bxr7_yUdpG49BoGPbuSbCQn3Rt/view?usp=sharing) | [CIFAR10 (VP SDE)](https://drive.google.com/file/d/1N82xJ7ZcPGZ45EUnxa7dOO36kJ2tMEtl/view?usp=sharing) | `"--method pred_eps_epsc_pretrained_ct --sde vpsde"` |
|[CIFAR10 (VP SDE), SN-DPM](https://drive.google.com/file/d/1NEqo4H902iF_N3777sWG0TKh3UlpxUZS/view?usp=sharing) | [CIFAR10 (VP SDE)](https://drive.google.com/file/d/1N82xJ7ZcPGZ45EUnxa7dOO36kJ2tMEtl/view?usp=sharing) | `"--method pred_eps_eps2_pretrained_ct --sde vpsde"` |
|[CelebA 64x64, NPR-DPM](https://drive.google.com/file/d/1IIIZ34FJK8tIQcj9sicaYQgJhrAaXLO-/view?usp=sharing) | [CelebA 64x64](https://drive.google.com/file/d/1bGQGTsFOnqQ2z3FN5rdkj1FPN1_5nYF4/view?usp=sharing) | `"--method pred_eps_epsc_pretrained"` |
|[CelebA 64x64, SN-DPM](https://drive.google.com/file/d/1ZHiC5RrW7XFeEdqeAiAXAV0Srp32BnVB/view?usp=sharing) | [CelebA 64x64](https://drive.google.com/file/d/1bGQGTsFOnqQ2z3FN5rdkj1FPN1_5nYF4/view?usp=sharing) | `"--method pred_eps_eps2_pretrained"` |
|[ImageNet 64x64, NPR-DPM](https://drive.google.com/file/d/1joSr575-TcMGXG3qmm72CBhH-QyTFalv/view?usp=sharing) | [ImageNet 64x64](https://drive.google.com/file/d/1evlXbMOg55y2BIjiALcD6Smbm07k7XGW/view?usp=sharing) | `"--method pred_eps_epsc_pretrained --mode simple"` |
|[ImageNet 64x64, SN-DPM](https://drive.google.com/file/d/1SC5_4nfIqL0cw6zbM-Ppfu8qFK-9vSoc/view?usp=sharing) | [ImageNet 64x64](https://drive.google.com/file/d/1evlXbMOg55y2BIjiALcD6Smbm07k7XGW/view?usp=sharing) | `"--method pred_eps_eps2_pretrained --mode complex"` |
|[LSUN Bedroom, NPR-DPM](https://drive.google.com/file/d/1JR9X-S_UN5FXWWv6fYRV8sLJpZ30fD3p/view?usp=sharing) | [LSUN Bedroom](https://drive.google.com/file/d/1fVxn3C5uaXdZM4cc8WnQ6GXexS5-274k/view?usp=sharing) | `"--method pred_eps_epsc_pretrained --mode simple"` |
|[LSUN Bedroom, SN-DPM](https://drive.google.com/file/d/1caFLEvCXXCDWDjWpL0wuPNCJ9wNqt5ly/view?usp=sharing) | [LSUN Bedroom](https://drive.google.com/file/d/1fVxn3C5uaXdZM4cc8WnQ6GXexS5-274k/view?usp=sharing) | `"--method pred_eps_eps2_pretrained --mode complex"` |

Here is the list of pretrained DPMs, collected from prior works. They are converted to a format that can be directly used for this codebase.
<a id="pretrained_dpm"/>

| Pretrained DPM | Expected mean squared norm (`ms_eps`) <br> (Used in Analytic-DPM)  | From |
|:----:|:----:|:----:|
| [CIFAR10 (LS)](https://drive.google.com/file/d/1rhZBWUDK3_q37Iac3sXq6WnxR_OHhPyI/view?usp=sharing) | [Link](https://drive.google.com/file/d/1V9XqMKxl3rEdjCYYHL5dwseA4QheHmbv/view?usp=sharing) | [Analytic-DPM](https://github.com/baofff/Analytic-DPM) |
| [CIFAR10 (CS)](https://drive.google.com/file/d/1ONNLpqPDLr4NesC0TfVZ3dCyaVBu7Xw0/view?usp=sharing) | [Link](https://drive.google.com/file/d/1cE5EgRHnCjmrfsXTOTGwNkWvcI18K6S2/view?usp=sharing) | [Analytic-DPM](https://github.com/baofff/Analytic-DPM) |
| [CIFAR10 (VP SDE)](https://drive.google.com/file/d/1N82xJ7ZcPGZ45EUnxa7dOO36kJ2tMEtl/view?usp=sharing) | [Link](https://drive.google.com/file/d/1cTp-iWDAumo6NidX9uCSPnUU7qIAch64/view?usp=sharing) | [score-sde](https://github.com/yang-song/score_sde_pytorch) |
| [CelebA 64x64](https://drive.google.com/file/d/1bGQGTsFOnqQ2z3FN5rdkj1FPN1_5nYF4/view?usp=sharing) | [Link](https://drive.google.com/file/d/1bsE9llJT5KXyGi74taqwfG2ubrKAyBJC/view?usp=sharing) | [DDIM](https://github.com/ermongroup/ddim) |
| [ImageNet 64x64](https://drive.google.com/file/d/1evlXbMOg55y2BIjiALcD6Smbm07k7XGW/view?usp=sharing) | [Link](https://drive.google.com/file/d/1M_exUFQMeXGRj1p2tJbE7vIMJBmmR9XR/view?usp=sharing) | [Improved DDPM](https://github.com/openai/improved-diffusion) |
| [LSUN Bedroom](https://drive.google.com/file/d/1fVxn3C5uaXdZM4cc8WnQ6GXexS5-274k/view?usp=sharing) | [Link](https://drive.google.com/file/d/1F7zCFxC912-wOsNsDIc4q4QRq_ZxII4p/view?usp=sharing) | [pytorch_diffusion](https://github.com/pesser/pytorch_diffusion) |

This [link](https://drive.google.com/drive/folders/1aqSXiJSFRqtqHBAsgUw4puZcRqrqOoHx?usp=sharing) provides precalculated FID statistics on CIFAR10, CelebA 64x64, ImageNet 64x64 and LSUN Bedroom. They are computed following Appendix F.2 in [Analytic-DPM](https://arxiv.org/abs/2201.06503).



## Evaluation Hyperparamters for NPR/SN-DPM and Analytic-DPM
<a id="evaluation"/>

**Note:** Analytic-DPM needs to precalculate the expected mean squared norm of noise prediction model (`ms_eps`), which is provided <a href="#pretrained_dpm">here</a>. Specify their path by `--ms_eps_path`.


* Sampling experiments on CIFAR10 (LS) or CelebA 64x64, Table 1 in the paper:

|  |`method` | `eval_hparams` |
|:----:|:----:|:----:|
|NPR-DDPM | `pred_eps_epsc_pretrained` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 2"` |
|SN-DDPM | `pred_eps_eps2_pretrained` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 2"` |
|Analytic-DDPM | `pred_eps` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 2 --ms_eps_path ms_eps_path"` |
|NPR-DDIM | `pred_eps_epsc_pretrained` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0"` |
|SN-DDIM | `pred_eps_eps2_pretrained` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0"` |
|Analytic-DDIM | `pred_eps` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0 --ms_eps_path ms_eps_path"` |


* Sampling experiments on CIFAR10 (CS), Table 1 in the paper:

|  |`method` | `eval_hparams` |
|:----:|:----:|:----:|
|NPR-DDPM | `pred_eps_epsc_pretrained` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --schedule cosine_1000"` |
|SN-DDPM | `pred_eps_eps2_pretrained` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --schedule cosine_1000"` |
|Analytic-DDPM | `pred_eps` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --schedule cosine_1000 --ms_eps_path ms_eps_path"` |
|NPR-DDIM | `pred_eps_epsc_pretrained` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0 --schedule cosine_1000"` |
|SN-DDIM | `pred_eps_eps2_pretrained` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0 --schedule cosine_1000"` |
|Analytic-DDIM | `pred_eps` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0 --schedule cosine_1000 --ms_eps_path ms_eps_path"` |


* Sampling experiments on CIFAR10 (VP SDE), Table 1 in the paper:

|  |`method` | `eval_hparams` |
|:----:|:----:|:----:|
|NPR-DDPM | `pred_eps_epsc_pretrained_ct2dt` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 2 --schedule vpsde_1000"` |
|SN-DDPM | `pred_eps_eps2_pretrained_ct2dt` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 2 --schedule vpsde_1000"` |
|Analytic-DDPM | `pred_eps_ct2dt` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 2 --schedule vpsde_1000 --ms_eps_path ms_eps_path"` |
|NPR-DDIM | `pred_eps_epsc_pretrained_ct2dt` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0 --schedule vpsde_1000"` |
|SN-DDIM | `pred_eps_eps2_pretrained_ct2dt` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0 --schedule vpsde_1000"` |
|Analytic-DDIM | `pred_eps_ct2dt` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0 --schedule vpsde_1000 --ms_eps_path ms_eps_path"` |


* Sampling experiments on ImageNet 64x64, Table 1 in the paper:

|  |`method` | `eval_hparams` |
|:----:|:----:|:----:|
|NPR-DDPM | `pred_eps_epsc_pretrained` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --mode simple"` |
|SN-DDPM | `pred_eps_eps2_pretrained` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --mode complex"` |
|Analytic-DDPM | `pred_eps` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --ms_eps_path ms_eps_path"` |
|NPR-DDIM | `pred_eps_epsc_pretrained` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0 --mode simple"` |
|SN-DDIM | `pred_eps_eps2_pretrained` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0 --mode complex"` |
|Analytic-DDIM | `pred_eps` | `"--rev_var_type optimal --clip_sigma_idx 1 --clip_pixel 1 --forward_type ddim --eta 0 --ms_eps_path ms_eps_path"` |



* Likelihood experiments on CIFAR10 (LS) or CelebA 64x64, Table 3 in the paper:

|  |`method` | `eval_hparams` |
|:----:|:----:|:----:|
|NPR-DDPM | `pred_eps_epsc_pretrained` | `"--rev_var_type optimal"` |
|Analytic-DDPM | `pred_eps` | `"--rev_var_type optimal --ms_eps_path ms_eps_path"` |


* Likelihood experiments on CIFAR10 (CS), Table 3 in the paper:

|  |`method` | `eval_hparams` |
|:----:|:----:|:----:|
|NPR-DDPM | `pred_eps_epsc_pretrained` | `"--rev_var_type optimal --schedule cosine_1000"` |
|Analytic-DDPM | `pred_eps` | `"--rev_var_type optimal --schedule cosine_1000 --ms_eps_path ms_eps_path"` |


* Likelihood experiments on ImageNet 64x64, Table 3 in the paper:

|  |`method` | `eval_hparams` |
|:----:|:----:|:----:|
|NPR-DDPM | `pred_eps_epsc_pretrained` | `"--rev_var_type optimal --mode simple"` |
|Analytic-DDPM | `pred_eps` | `"--rev_var_type optimal --ms_eps_path ms_eps_path"` |



## This implementation is based on / inspired by

* [Analytic-DPM](https://github.com/baofff/Analytic-DPM) (provide the code structure)

* [pytorch_diffusion](https://github.com/pesser/pytorch_diffusion) (provide codes of models for CelebA64x64 and LSUN Bedroom)

* [Improved DDPM](https://github.com/openai/improved-diffusion) (provide codes of models for CIFAR10 and Imagenet64x64)

* [score-sde](https://github.com/yang-song/score_sde_pytorch)  (provide codes of models for CIFAR10)

* [pytorch-fid](https://github.com/mseitzer/pytorch-fid) (provide the official implementation of FID to PyTorch)
