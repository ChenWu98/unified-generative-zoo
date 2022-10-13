# Created by Chen Henry Wu
import os
import sys
sys.path.append(os.path.abspath('model/lib/diffusion_stylegan'))
import torch
import torchvision.transforms as transforms

from dnnlib.util import open_url
from legacy import load_network_pkl
from ..model_utils import requires_grad


def prepare_diffusion_stylegan2(source_model_type):
    pt_file_name = {
        "ffhq":      "diffusion-stylegan2-ffhq.pkl",
    }[source_model_type]
    dsg2ckpt = os.path.join('ckpts', pt_file_name)

    return dsg2ckpt


class DiffusionStyleGAN2Wrapper(torch.nn.Module):

    def __init__(self, source_model_type, sample_truncation):
        super(DiffusionStyleGAN2Wrapper, self).__init__()

        self.sample_truncation = sample_truncation

        # Set up generator
        self.dsg2ckpt = prepare_diffusion_stylegan2(source_model_type)
        with open_url(self.dsg2ckpt) as f:
            self.generator = load_network_pkl(f)['G_ema']
        self.latent_dim = self.generator.z_dim
        assert self.generator.c_dim == 0

        # Freeze.
        requires_grad(self.generator, False)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

    def forward(self, z):
        # Eval mode for the generator.
        self.generator.eval()

        img = self.generator(z=z, c=None, truncation_psi=1 if self.training else self.sample_truncation)
        # Post process.
        img = self.post_process(img)

        return img

    @property
    def device(self):
        return next(self.parameters()).device

