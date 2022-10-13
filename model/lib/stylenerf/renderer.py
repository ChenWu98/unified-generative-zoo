# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""Wrap the generator to render a sequence of images"""
import torch
import torch.nn.functional as F
import numpy as np
from torch import random
import tqdm
import copy
import trimesh


class Renderer(object):

    def __init__(self, generator, discriminator=None, program=None):
        self.generator = generator
        self.discriminator = discriminator
        self.sample_tmp = 0.65
        self.program = program
        self.seed = 0

        if (program is not None) and (len(program.split(':')) == 2):
            from training.dataset import ImageFolderDataset
            self.image_data = ImageFolderDataset(program.split(':')[1])
            self.program = program.split(':')[0]
        else:
            self.image_data = None

    def set_random_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def __call__(self, *args, **kwargs):
        self.generator.eval()  # eval mode...

        if self.program is None:
            if hasattr(self.generator, 'get_final_output'):
                return self.generator.get_final_output(*args, **kwargs)
            return self.generator(*args, **kwargs)
        
        if self.image_data is not None:
            batch_size = 1
            indices = (np.random.rand(batch_size) * len(self.image_data)).tolist()
            rimages = np.stack([self.image_data._load_raw_image(int(i)) for i in indices], 0)
            rimages = torch.from_numpy(rimages).float().to(kwargs['z'].device) / 127.5 - 1
            kwargs['img'] = rimages
        
        outputs = getattr(self, f"render_{self.program}")(*args, **kwargs)
        
        if self.image_data is not None:
            imgs = outputs if not isinstance(outputs, tuple) else outputs[0]
            size = imgs[0].size(-1)
            rimg = F.interpolate(rimages, (size, size), mode='bicubic', align_corners=False)
            imgs = [torch.cat([img, rimg], 0) for img in imgs]
            outputs = imgs if not isinstance(outputs, tuple) else (imgs, outputs[1])
        return outputs

    def get_additional_params(self, ws, t=0):
        gen = self.generator.synthesis
        batch_size = ws.size(0)

        kwargs = {}
        if not hasattr(gen, 'get_latent_codes'):
            return kwargs

        s_val, t_val, r_val = [[0, 0, 0]], [[0.5, 0.5, 0.5]], [0.]
        # kwargs["transformations"] = gen.get_transformations(batch_size=batch_size, mode=[s_val, t_val, r_val], device=ws.device)
        # kwargs["bg_rotation"] = gen.get_bg_rotation(batch_size, device=ws.device)
        # kwargs["light_dir"] = gen.get_light_dir(batch_size, device=ws.device)
        kwargs["latent_codes"] = gen.get_latent_codes(batch_size, tmp=self.sample_tmp, device=ws.device)
        kwargs["camera_matrices"] = self.get_camera_traj(t, ws.size(0), device=ws.device)
        return kwargs

    def get_camera_traj(self, t, batch_size=1, traj_type='pigan', device='cpu'):
        gen = self.generator.synthesis
        if traj_type == 'pigan':
            range_u, range_v = gen.C.range_u, gen.C.range_v
            pitch = 0.2 * np.cos(t * 2 * np.pi) + np.pi/2
            yaw = 0.4 * np.sin(t * 2 * np.pi)
            u = (yaw - range_u[0]) / (range_u[1] - range_u[0])
            v = (pitch - range_v[0]) / (range_v[1] - range_v[0])
            cam = gen.get_camera(batch_size=batch_size, mode=[u, v, 0.5], device=device)
        else:
            raise NotImplementedError
        return cam
