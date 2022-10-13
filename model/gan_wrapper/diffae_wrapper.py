# Created by Chen Henry Wu
import os
import torch
import torchvision.transforms as transforms

from ..lib.diffae.templates_latent import (
    ffhq128_autoenc_latent,
    ffhq256_autoenc_latent,
    horse128_autoenc_latent,
    bedroom128_autoenc_latent,
    LitModel,
)
from ..lib.diffae.config import TrainConfig, Sampler, BeatGANsAutoencModel
from ..model_utils import requires_grad


def prepare_diffae(source_model_type):
    pt_file_name = {
        "ffhq128":      "diffae_ffhq128.ckpt",
        "ffhq256":      "diffae_ffhq256.ckpt",
        "horse128":     "diffae_horse128.ckpt",
        "bedroom128":   "diffae_bedroom128.ckpt",
    }[source_model_type]
    diffae_ckpt = os.path.join('ckpts', pt_file_name)

    latent_file_name = {
        "ffhq128":      "diffae_ffhq128_latent.pkl",
        "ffhq256":      "diffae_ffhq256_latent.pkl",
        "horse128":     "diffae_horse128_latent.pkl",
        "bedroom128":   "diffae_bedroom128_latent.pkl",
    }[source_model_type]
    diffae_latent = os.path.join('ckpts', latent_file_name)

    conf = {
        "ffhq128":      ffhq128_autoenc_latent,
        "ffhq256":      ffhq256_autoenc_latent,
        "horse128":     horse128_autoenc_latent,
        "bedroom128":   bedroom128_autoenc_latent,
    }[source_model_type]()

    conf.T_eval = 100
    conf.latent_T_eval = 100
    conf.pretrain.path = diffae_ckpt
    conf.latent_infer_path = diffae_latent
    print(conf)

    return conf


def render_uncondition(latent_noise,
                       conf: TrainConfig,
                       model: BeatGANsAutoencModel,
                       x_T,
                       sampler: Sampler,
                       latent_sampler: Sampler,
                       conds_mean=None,
                       conds_std=None,
                       clip_latent_noise: bool = False):

    if clip_latent_noise:
        latent_noise = latent_noise.clip(-1, 1)

    cond = latent_sampler.sample(
        model=model.latent_net,
        noise=latent_noise,
        clip_denoised=conf.latent_clip_sample,
    )

    if conf.latent_znormalize:
        cond = cond * conds_std + conds_mean

    # the diffusion on the model
    return sampler.sample(model=model, noise=x_T, cond=cond)


class DiffAEWrapper(torch.nn.Module):

    def __init__(self, source_model_type, latent_only, steps_train, latent_steps_train):
        super(DiffAEWrapper, self).__init__()

        self.latent_only = latent_only
        self.steps_train = steps_train
        self.latent_steps_train = latent_steps_train

        # Set up generator
        self.conf = prepare_diffae(source_model_type)
        self.generator = LitModel(self.conf)
        if self.latent_only:
            self.latent_dim = self.generator.conf.style_ch
        else:
            self.latent_dim = (
                self.generator.conf.style_ch +
                self.generator.conf.img_size ** 2 * self.generator.conf.model_conf.out_channels
            )
        # Freeze.
        requires_grad(self.generator, False)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

    def generate(self, z):
        if not self.training:
            sampler = self.generator.eval_sampler
            latent_sampler = self.generator.latent_sampler
        else:
            T = self.steps_train
            T_latent = self.latent_steps_train
            assert T is not None and T_latent is not None
            sampler = self.generator.conf._make_diffusion_conf(T).make_sampler()
            latent_sampler = self.generator.conf._make_latent_diffusion_conf(T_latent).make_sampler()
        #print('sampler.conf.betas:', sampler.conf.betas)
        #print('latent_sampler.conf.betas:', latent_sampler.conf.betas)

        if self.latent_only:
            latent = z
            noise = torch.randn(z.shape[0],
                                self.generator.conf.model_conf.out_channels,
                                self.generator.conf.img_size,
                                self.generator.conf.img_size,
                                device=self.device)
        else:
            latent = z[:, :self.generator.conf.style_ch]
            noise = z[:, self.generator.conf.style_ch:].view(
                z.shape[0],
                self.generator.conf.model_conf.out_channels,
                self.generator.conf.img_size,
                self.generator.conf.img_size,
            )

        img = render_uncondition(
            latent,
            self.generator.conf,
            self.generator.ema_model,
            noise,
            sampler=sampler,
            latent_sampler=latent_sampler,
            conds_mean=self.generator.conds_mean,
            conds_std=self.generator.conds_std,
        )

        return img

    def forward(self, z):
        # Eval mode for the generator.
        self.generator.eval()

        img = self.generate(z)

        # Post process.
        img = self.post_process(img)

        return img

    @property
    def device(self):
        return next(self.parameters()).device




