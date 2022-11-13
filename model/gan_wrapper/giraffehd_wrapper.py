import argparse

import numpy as np
import torch
import torchvision.transforms as transforms

from ..lib.giraffe_hd.model import GIRAFFEHDGenerator
from ..model_utils import requires_grad


def prepare_ghq(source_model_type):
    print('First of all, when the code changes, make sure that no part in the model is under no_grad!')

    parser = argparse.ArgumentParser(description="Giraffe trainer")

    parser.add_argument('--inj_idx', type=int, default=-1, help='inject index for evaluation')

    if source_model_type == 'ffhq1024':
        ckpt = 'ckpts/giraffehd_ffhq_1024.pt'
        config = parser.parse_args([])
    else:
        raise ValueError("Unknown source model type: {}".format(source_model_type))

    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    config.ckpt_args = ckpt['args']
    if config.inj_idx == -1:
        if config.ckpt_args.size == 256:
            config.inj_idx = 2
        elif config.ckpt_args.size == 512:
            config.inj_idx = 4
        elif config.ckpt_args.size == 1024:
            config.inj_idx = 4

    print('config:', config)

    return ckpt, config


class GIRAFFEHDWrapper(torch.nn.Module):

    def __init__(self, source_model_type, temperature):
        super(GIRAFFEHDWrapper, self).__init__()

        self.temperature = temperature

        # Set up generator
        ckpt, self.config = prepare_ghq(source_model_type)
        self.generator = GIRAFFEHDGenerator(
            device='cuda',
            z_dim=self.config.ckpt_args.z_dim,
            z_dim_bg=self.config.ckpt_args.z_dim_bg,
            size=self.config.ckpt_args.size,
            resolution_vol=self.config.ckpt_args.res_vol,
            feat_dim=self.config.ckpt_args.feat_dim,
            range_u=self.config.ckpt_args.range_u,
            range_v=self.config.ckpt_args.range_v,
            fov=self.config.ckpt_args.fov,
            scale_range_max=self.config.ckpt_args.scale_range_max,
            scale_range_min=self.config.ckpt_args.scale_range_min,
            translation_range_max=self.config.ckpt_args.translation_range_max,
            translation_range_min=self.config.ckpt_args.translation_range_min,
            rotation_range=self.config.ckpt_args.rotation_range,
            bg_translation_range_max=self.config.ckpt_args.bg_translation_range_max,
            bg_translation_range_min=self.config.ckpt_args.bg_translation_range_min,
            bg_rotation_range=self.config.ckpt_args.bg_rotation_range,
            refine_n_styledconv=2,
            refine_kernal_size=3,
            grf_use_mlp=self.config.ckpt_args.grf_use_mlp,
            pos_share=self.config.ckpt_args.pos_share,
            use_viewdirs=self.config.ckpt_args.use_viewdirs,
            grf_use_z_app=self.config.ckpt_args.grf_use_z_app,
            fg_gen_mask=self.config.ckpt_args.fg_gen_mask
        )
        self.generator.load_state_dict(ckpt["g_ema"])

        self.z_dim_obj = self.generator.vol_generator.z_dim
        self.z_dim_bg = self.generator.vol_generator.z_dim_bg
        self.latent_dim = self.z_dim_obj * 2 + self.z_dim_bg * 2

        # Freeze.
        requires_grad(self.generator, False)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

    def forward(self, z):
        # Eval mode for the generator.
        self.generator.eval()

        z_shape_obj = z[:, :self.z_dim_obj].unsqueeze(1) * self.temperature
        z_app_obj = z[:, self.z_dim_obj:self.z_dim_obj * 2].unsqueeze(1) * self.temperature
        z_shape_bg = z[:, self.z_dim_obj * 2:self.z_dim_obj * 2 + self.z_dim_bg] * self.temperature
        z_app_bg = z[:, self.z_dim_obj * 2 + self.z_dim_bg:] * self.temperature

        img_rep = self.generator.get_rep_from_code(latent_codes=(z_shape_obj, z_app_obj, z_shape_bg, z_app_bg))
        img = self.generator(img_rep=img_rep, inject_index=self.config.inj_idx, mode='eval')[0]

        # Post process.
        img = self.post_process(img)

        return img

    @property
    def device(self):
        return next(self.parameters()).device




