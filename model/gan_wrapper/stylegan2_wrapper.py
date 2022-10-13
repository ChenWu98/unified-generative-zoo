# Created by Chen Henry Wu
import os
import torch
import torchvision.transforms as transforms

from ..lib.stylegan2.sg2_model import Generator
from ..model_utils import requires_grad


def prepare_stylegan(source_model_type):
    pt_file_name = {
        "ffhq":      "ffhq.pt",
        "cat":       "afhqcat.pt",
        "dog":       "afhqdog.pt",
        "church":    "stylegan2-church-config-f.pt",
        "car":       "stylegan2-car-config-f.pt",
        "horse":     "stylegan2-horse-config-f.pt",
        "wild":      "wild.pt",
        "metfaces":  "metfaces.pt",
        "brecahad":  "brecahad.pt",
        "landscape": "landscape.pt",
    }[source_model_type]
    sg2ckpt = os.path.join('ckpts', pt_file_name)

    size = {
        "ffhq":      1024,
        "cat":       512,
        "dog":       512,
        "church":    256,
        "horse":     256,
        "car":       512,
        "wild":      512,
        "metfaces":  1024,
        "brecahad":  512,
        "landscape": 256,
    }[source_model_type]

    channel_multiplier = 2

    return sg2ckpt, size, channel_multiplier


class StyleGAN2Wrapper(torch.nn.Module):

    def __init__(self, source_model_type, sample_truncation, randomize_noise):
        super(StyleGAN2Wrapper, self).__init__()

        self.sample_truncation = sample_truncation
        self.randomize_noise = randomize_noise

        # Set up generator
        self.latent_dim = 512
        self.sg2ckpt, self.img_size, self.channel_multiplier = prepare_stylegan(source_model_type)
        self.generator = Generator(size=self.img_size,
                                   style_dim=self.latent_dim,
                                   n_mlp=8,
                                   channel_multiplier=self.channel_multiplier)
        self.generator.load_state_dict(torch.load(self.sg2ckpt)["g_ema"], strict=True)
        # Freeze.
        requires_grad(self.generator, False)

        # Register mean of W
        self.register_mean()

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

    def register_mean(self):
        # Eval mode.
        self.generator.eval()

        with torch.no_grad():
            # Eval mode.
            latents = self.generator.style(
                torch.randn(
                    50000, self.generator.style_dim
                )
            )

            # Vanilla latents.
            mean_latent = latents.mean(0)
            self.register_buffer('mean_latent', mean_latent)

    def z_to_w(self, z):
        assert z.dim() == 2
        w = self.generator.style(z)

        return w

    def forward(self, z):
        # Eval mode for the generator.
        self.generator.eval()

        w = self.z_to_w(z)

        img = self.generator([w],
                             input_is_latent=True,  # True if w; False if z
                             truncation=1 if self.training else self.sample_truncation,
                             truncation_latent=self.mean_latent,
                             randomize_noise=self.randomize_noise)[0]
        # Post process.
        img = self.post_process(img)

        return img

    @property
    def device(self):
        return next(self.parameters()).device




