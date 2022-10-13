# Created by Chen Henry Wu
import os
import sys
sys.path.append(os.path.abspath('model/lib/eg3d'))
import torch
import torchvision.transforms as transforms
import numpy as np

from dnnlib.util import open_url
from legacy import load_network_pkl
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

from ..model_utils import requires_grad


def prepare_eg3d(source_model_type):
    sw2ckpt = os.path.join('ckpts', f"{source_model_type}.pkl")

    return sw2ckpt


class EG3DWrapper(torch.nn.Module):

    def __init__(self, source_model_type, sample_truncation):
        super(EG3DWrapper, self).__init__()

        self.sample_truncation = sample_truncation

        # Set up generator
        self.latent_dim = 512  # TODO
        self.eg3dckpt = prepare_eg3d(source_model_type)
        with open_url(self.eg3dckpt) as f:
            self.generator = load_network_pkl(f)['G_ema']

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

        # Following the default setting in gen_samples.py of EG3D.
        fov_deg = 18.837
        angle_y = 0
        angle_p = -0.2

        intrinsics = FOV_to_intrinsics(fov_deg, device=self.device)
        cam_pivot = torch.tensor(self.generator.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=self.device)
        cam_radius = self.generator.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot,
                                                  radius=cam_radius, device=self.device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius,
                                                               device=self.device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        ws = self.generator.mapping(z,
                                    conditioning_params,
                                    truncation_psi=1 if self.training else self.sample_truncation)
        img = self.generator.synthesis(ws, camera_params)['image']

        # Post process.
        img = self.post_process(img)

        return img

    @property
    def device(self):
        return next(self.parameters()).device




