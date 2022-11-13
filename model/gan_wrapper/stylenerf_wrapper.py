import os
import sys
sys.path.append(os.path.abspath('model/lib/stylenerf'))
import torch
import torchvision.transforms as transforms

from dnnlib.util import open_url
from legacy import load_network_pkl
from renderer import Renderer
from ..model_utils import requires_grad


def prepare_stylenerf(source_model_type):
    pt_file_name = {
        "ffhq256":  "StyleNeRF_ffhq_256.pkl",
        "ffhq512":  "StyleNeRF_ffhq_512.pkl",
        "ffhq1024": "StyleNeRF_ffhq_1024.pkl",
    }[source_model_type]
    sw2ckpt = os.path.join('ckpts', pt_file_name)

    return sw2ckpt


class StyleNeRFWrapper(torch.nn.Module):

    def __init__(self, source_model_type, sample_truncation):
        super(StyleNeRFWrapper, self).__init__()

        self.sample_truncation = sample_truncation

        # Set up generator
        self.latent_dim = 512  # TODO
        self.snckpt = prepare_stylenerf(source_model_type)
        with open_url(self.snckpt) as f:
            G = load_network_pkl(f)['G_ema']
            assert G.c_dim == 0

        # avoid persistent classes...
        from training.networks import Generator
        from torch_utils import misc
        with torch.no_grad():
            self.generator = Generator(*G.init_args, **G.init_kwargs)
            misc.copy_params_and_buffers(G, self.generator, require_all=False)
        self.renderer = Renderer(self.generator, None, None)

        # Freeze.
        requires_grad(self.generator, False)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

    def forward(self, z):
        # Eval mode for the generator.
        self.generator.eval()

        assert z.dim() == 2

        relative_range_u_scale = 1.0
        relative_range_u = [0.5 - 0.5 * relative_range_u_scale, 0.5 + 0.5 * relative_range_u_scale]
        outputs = self.renderer(
            z=z,
            c=None,
            truncation_psi=1 if self.training else self.sample_truncation,
            noise_mode='random',  # 'const', 'random', 'none'
            render_option=None,
            n_steps=1,
            relative_range_u=relative_range_u,
            return_cameras=True,
        )
        if isinstance(outputs, tuple):
            img, cameras = outputs
        else:
            img = outputs

        # Post process.
        img = self.post_process(img)

        return img

    @property
    def device(self):
        return next(self.parameters()).device




