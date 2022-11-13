import os
import torch
import torchvision.transforms as transforms

from ..lib.styleswin.models.generator import Generator
from ..model_utils import requires_grad


def prepare_styleswin(source_model_type):
    pt_file_name = {
        "ffhq256":      "StyleSwin_FFHQ_256.pt",
        "ffhq1024":     "StyleSwin_FFHQ_1024.pt",
        "celebahq256":  "StyleSwin_CelebAHQ_256.pt",
        "celebahq1024": "StyleSwin_CelebAHQ_1024.pt",
        "church":       "StyleSwin_LSUNChurch_256.pt",
    }[source_model_type]
    sw2ckpt = os.path.join('ckpts', pt_file_name)

    size = {
        "ffhq256":      256,
        "ffhq1024":     1024,
        "celebahq256":  256,
        "celebahq1024": 1024,
        "church":       256,
    }[source_model_type]

    if size == 256:
        channel_multiplier = 2
    elif size == 1024:
        channel_multiplier = 1
    else:
        raise ValueError()

    return sw2ckpt, size, channel_multiplier


class StyleSwinWrapper(torch.nn.Module):

    def __init__(self, source_model_type, sample_truncation):
        super(StyleSwinWrapper, self).__init__()

        self.sample_truncation = sample_truncation

        # Set up generator
        self.latent_dim = 512
        self.swckpt, self.img_size, self.channel_multiplier = prepare_styleswin(source_model_type)
        self.generator = Generator(size=self.img_size,
                                   style_dim=self.latent_dim,
                                   n_mlp=8,
                                   channel_multiplier=self.channel_multiplier)
        self.generator.load_state_dict(torch.load(self.swckpt, map_location='cpu')["g_ema"], strict=True)
        # Freeze.
        requires_grad(self.generator, False)

        # Register mean of W
        self.register_mean()

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]-normalized (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])]
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

    def forward(self, z):
        # Eval mode for the generator.
        self.generator.eval()

        assert z.dim() == 2
        img = self.generator([z],
                             truncation=1 if self.training else self.sample_truncation,
                             truncation_latent=self.mean_latent)[0]
        # Post process.
        img = self.post_process(img)

        return img

    @property
    def device(self):
        return next(self.parameters()).device




