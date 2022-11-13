import os
import sys
sys.path.append(os.path.abspath('model/lib/stylegan_xl'))
import torch
import torchvision.transforms as transforms

from dnnlib.util import open_url
from legacy import load_network_pkl
from ..model_utils import requires_grad


class StyleGANXLWrapper(torch.nn.Module):

    def __init__(self, network_pkl, sample_truncation):
        super(StyleGANXLWrapper, self).__init__()

        self.sample_truncation = sample_truncation

        # Set up generator
        with open_url(network_pkl) as f:
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

