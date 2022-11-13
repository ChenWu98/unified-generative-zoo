import os
import numpy as np
import torch
from torch.cuda.amp import autocast

from ..lib.nvae.model import AutoEncoder
from ..lib.nvae.utils import get_arch_cells
from ..lib.nvae.distributions import Normal, NormalDecoder, DiscMixLogistic
from ..model_utils import requires_grad


def prepare_nvae(source_model_type):
    pt_file_name = {
        "ffhq":      "nvae_ffhq_256.pt",
        "celebahq":  "nvae_celebahq_256.pt",
    }[source_model_type]
    nvaeckpt = os.path.join('ckpts', pt_file_name)

    return nvaeckpt


def prepare_nvae_args_official(checkpoint):
    args = checkpoint['args']
    if not hasattr(args, 'ada_groups'):
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        args.min_groups_per_scale = 1

    if not hasattr(args, 'num_mixture_dec'):
        args.num_mixture_dec = 10

    arch_instance = get_arch_cells(args.arch_instance)

    return args, arch_instance


def set_bn_official(model, bn_eval_mode, num_samples=1, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
    else:
        model.train()
        with autocast():
            for i in range(iter):
                if i % 10 == 0:
                    print('setting BN statistics iter %d out of %d' % (i+1, iter))
                model.sample(num_samples, t)
        model.eval()


class NVAETruncWrapper(torch.nn.Module):

    def __init__(self, source_model_type, temperature, control_groups):
        super(NVAETruncWrapper, self).__init__()

        self.temp = temperature
        self.control_groups = control_groups

        # Set up generator
        self.nvaeckpt = prepare_nvae(source_model_type)
        checkpoint = torch.load(self.nvaeckpt, map_location='cpu')
        nvae_args, nvae_arch_instance = prepare_nvae_args_official(checkpoint)
        self.generator = AutoEncoder(nvae_args, None, nvae_arch_instance)
        # A quote from the official repo:
        # Loading is not strict because of self.weight_normalized in Conv2D class in neural_operations. This variable
        # is only used for computing the spectral normalization and it is safe not to load it. Some of our earlier models
        # did not have this variable.
        self.generator.load_state_dict(checkpoint['state_dict'], strict=False)
        self.generator.cuda()
        bn_eval_mode = False
        with torch.no_grad():
            set_bn_official(self.generator, bn_eval_mode, num_samples=16, t=self.temp, iter=500)

        # Freeze.
        requires_grad(self.generator, False)

        # Get latent dim.
        with torch.no_grad():
            self.latent_dim = self.get_latent_dim()
            print(f'Latent dim: {self.latent_dim}')

        # Post process.
        # No post process for NVAE, handled in the decoder distribution.

    def get_latent_dim(self):
        bsz = 1
        scale_ind = 0
        latent_dim = int(np.prod(self.generator.z0_size))
        z = torch.zeros(bsz, *self.generator.z0_size, device=self.device) * self.temp
        group_ind = 1

        idx_dec = 0
        s = self.generator.prior_ftr0.unsqueeze(0).expand(bsz, -1, -1, -1)
        for cell in self.generator.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    param = self.generator.dec_sampler[idx_dec - 1](s)
                    mu, log_sigma = torch.chunk(param, 2, dim=1)
                    if group_ind > self.control_groups - 1:
                        pass
                    else:
                        latent_dim += int(np.prod(mu.shape[1:]))
                    z = mu + log_sigma.exp() * torch.zeros_like(mu) * self.temp
                    group_ind += 1

                # 'combiner_dec'
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)
                if cell.cell_type == 'up_dec':
                    scale_ind += 1
        return latent_dim

    def generate(self, concat_z):
        bsz = concat_z.shape[0]
        scale_ind = 0
        z_pointer_prev = 0
        z_pointer = int(np.prod(self.generator.z0_size))
        z = concat_z[:, z_pointer_prev:z_pointer].view(bsz, *self.generator.z0_size) * self.temp
        group_ind = 1

        idx_dec = 0
        s = self.generator.prior_ftr0.unsqueeze(0).expand(bsz, -1, -1, -1)
        for cell in self.generator.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    param = self.generator.dec_sampler[idx_dec - 1](s)
                    mu, log_sigma = torch.chunk(param, 2, dim=1)
                    if z_pointer == concat_z.shape[1]:
                        assert group_ind > self.control_groups - 1
                        dist = Normal(mu, log_sigma, self.temp)
                        z, _ = dist.sample()
                    else:
                        z_pointer_prev = z_pointer
                        z_pointer += int(np.prod(mu.shape[1:]))
                        z = mu + log_sigma.exp() * concat_z[:, z_pointer_prev:z_pointer].view(*mu.shape) * self.temp
                        group_ind += 1

                # 'combiner_dec'
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)
                if cell.cell_type == 'up_dec':
                    scale_ind += 1

        assert z_pointer == concat_z.shape[1], f'z_pointer: {z_pointer}, concat_z.shape[1]: {concat_z.shape[1]}'
        assert not self.generator.vanilla_vae

        for cell in self.generator.post_process:
            s = cell(s)

        logits = self.generator.image_conditional(s)
        return logits

    def decode_output(self, logits):
        if self.generator.num_mix_output == 1:
            dist = NormalDecoder(logits, num_bits=self.generator.num_bits)
        else:
            dist = DiscMixLogistic(logits, self.generator.num_mix_output, num_bits=self.generator.num_bits)
        return dist.sample()

    def forward(self, z):
        # Eval mode for the generator.
        self.generator.eval()

        z = z.clamp(-1.25, 1.25)  # Truncate z.
        logits = self.generate(z)
        img = self.decode_output(logits)  # Already in [0, 1].

        # Post process.
        # No post process for NVAE, handled in the decoder distribution.

        return img

    @property
    def device(self):
        return next(self.parameters()).device




