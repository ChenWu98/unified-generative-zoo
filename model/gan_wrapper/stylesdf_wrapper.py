# Created by Chen Henry Wu
import os
import torch
import torchvision.transforms as transforms

from ..lib.stylesdf.options import BaseOptions
from ..lib.stylesdf.model import Generator
from ..lib.stylesdf.utils import generate_camera_params
from ..model_utils import requires_grad


def prepare_stylesdf(source_model_type, sample_truncation):

    opt = BaseOptions().parse()

    opt.experiment.expname = {
        "ffhq": "stylesdf_ffhq1024x1024",
        "afhq": "stylesdf_afhq512x512",
    }[source_model_type]
    opt.model.size = {
        "ffhq": 1024,
        "afhq": 512,
    }[source_model_type]

    opt.model.is_test = True
    opt.model.freeze_renderer = False
    opt.rendering.offset_sampling = True
    opt.rendering.static_viewdirs = True
    opt.rendering.force_background = True
    opt.rendering.perturb = 0
    opt.inference.size = opt.model.size
    opt.inference.camera = opt.camera
    opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.inference.style_dim = opt.model.style_dim
    opt.inference.project_noise = opt.model.project_noise
    opt.inference.return_xyz = opt.rendering.return_xyz
    opt.inference.truncation_ratio = sample_truncation

    return opt


class StyleSDFWrapper(torch.nn.Module):

    def __init__(self, source_model_type, sample_truncation):
        super(StyleSDFWrapper, self).__init__()

        # Set up generator
        self.opt = prepare_stylesdf(source_model_type, sample_truncation)
        self.latent_dim = self.opt.inference.style_dim
        checkpoint_path = os.path.join('ckpts', self.opt.experiment.expname + '.pt')
        self.generator = Generator(self.opt.model, self.opt.rendering)
        self.generator.load_state_dict(torch.load(checkpoint_path, map_location='cpu')["g_ema"], strict=True)

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
            self.generator.cuda()
            renderer_latent_mean, decoder_latent_mean = self.generator.mean_latent(self.opt.inference.truncation_mean, 'cuda')
            self.register_buffer('renderer_latent_mean', renderer_latent_mean)
            self.register_buffer('decoder_latent_mean', decoder_latent_mean)

    def forward(self, z):
        # Eval mode for the generator.
        self.generator.eval()

        opt = self.opt.inference
        locations = None
        fov = opt.camera.fov

        sample_cam_extrinsics, sample_focals, sample_near, sample_far, sample_locations = generate_camera_params(
            opt.renderer_output_size,
            self.device,
            batch=z.shape[0],
            locations=locations,
            uniform=opt.camera.uniform,
            azim_range=opt.camera.azim,
            elev_range=opt.camera.elev,
            fov_ang=fov,
            dist_radius=opt.camera.dist_radius
        )
        out = self.generator([z],
                             sample_cam_extrinsics,
                             sample_focals,
                             sample_near,
                             sample_far,
                             truncation=1 if self.training else opt.truncation_ratio,
                             truncation_latent=[self.renderer_latent_mean, self.decoder_latent_mean])
        img = out[0]

        # Post process.
        img = self.post_process(img)

        return img

    @property
    def device(self):
        return next(self.parameters()).device
