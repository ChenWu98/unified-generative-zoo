import argparse

import numpy as np
import torch
import torchvision.transforms as transforms

from ..lib.ddgan.score_sde.models.ncsnpp_generator_adagn import NCSNpp
from ..model_utils import requires_grad


def prepare_ddgan(source_model_type):
    print('First of all, when the code changes, make sure that no part in the model is under no_grad!')

    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # geenrator and training
    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)

    if source_model_type == 'celebahq256':
        ckpt = 'ckpts/ddgan_celebahq256_netG_550.pth'
        config = parser.parse_args(
            [
                '--image_size', str(256),
                '--num_channels', str(3),
                '--num_channels_dae', str(64),
                '--ch_mult', str(1), str(1), str(2), str(2), str(4), str(4),
                '--num_timesteps', str(2),
                '--num_res_blocks', str(2),
            ]
        )
        print('config:', config)
    else:
        raise ValueError("Unknown source model type: {}".format(source_model_type))

    return ckpt, config


def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Posterior_Coefficients:
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t, noise=None):
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t, noise=None):
        mean, _, log_var = q_posterior(x_0, x_t, torch.full((x_0.size(0),), t, dtype=torch.int64).to(x_0.device))

        if t == 0:
            return mean
        else:
            if noise is None:
                noise = torch.randn_like(x_t)

            return mean + torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t, noise=noise)

    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x_init, z_t_list, noise_list, opt):
    assert z_t_list.shape[1] <= n_time
    assert noise_list.shape[1] < n_time

    x = x_init
    for z_idx, t in enumerate(reversed(range(n_time))):

        t_time = torch.full((x.size(0),), t, dtype=torch.int64).to(x.device)
        if z_idx < z_t_list.shape[1]:
            latent_z = z_t_list[:, z_idx, :]
        else:
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)

        x_0 = generator(x, t_time, latent_z)
        if z_idx < noise_list.shape[1]:
            x_new = sample_posterior(coefficients, x_0, x, t, noise_list[:, z_idx, :, :, :])
        else:
            x_new = sample_posterior(coefficients, x_0, x, t)
        x = x_new

    return x


class DDGANWrapper(torch.nn.Module):

    def __init__(self, source_model_type, white_box_z_steps):
        super(DDGANWrapper, self).__init__()

        self.white_box_z_steps = white_box_z_steps

        # Set up generator
        ckpt, self.config = prepare_ddgan(source_model_type)
        self.generator = NCSNpp(self.config)
        state_dict = torch.load(ckpt, map_location='cpu')

        # Loading weights from ddp in single gpu
        for key in list(state_dict.keys()):
            state_dict[key[7:]] = state_dict.pop(key)
        self.generator.load_state_dict(state_dict)

        self.latent_dim = (
            self.config.image_size ** 2 * self.config.num_channels +
            self.white_box_z_steps * self.config.nz +
            (self.white_box_z_steps - 1) * self.config.image_size ** 2 * self.config.num_channels
        )

        # Freeze.
        requires_grad(self.generator, False)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

    def generate(self, z):
        bsz = z.shape[0]
        x_T = z[:, :self.config.image_size ** 2 * self.config.num_channels]
        x_T = x_T.view(
            bsz, self.config.num_channels, self.config.image_size, self.config.image_size
        )
        z = z[:, self.config.image_size ** 2 * self.config.num_channels:]

        z_t_list = []
        noise_list = []
        for t in range(self.white_box_z_steps):
            z_t = z[:, :self.config.nz]
            z_t_list.append(z_t)
            z = z[:, self.config.nz:]
            if t < self.white_box_z_steps - 1:
                noise_t = z[:, :self.config.image_size ** 2 * self.config.num_channels]
                noise_list.append(noise_t)
                z = z[:, self.config.image_size ** 2 * self.config.num_channels:]
        assert z.shape[1] == 0
        z_t_list = torch.stack(z_t_list, dim=1).view(
            bsz,
            self.white_box_z_steps,
            self.config.nz,
        )
        noise_list = torch.stack(noise_list, dim=1).view(
            bsz,
            self.white_box_z_steps - 1,
            self.config.num_channels,
            self.config.image_size,
            self.config.image_size,
        )

        pos_coeff = Posterior_Coefficients(self.config, self.device)
        img = sample_from_model(
            pos_coeff,
            self.generator,
            self.config.num_timesteps,
            x_T,
            z_t_list,
            noise_list,
            self.config
        )

        return img

    def forward(self, z):
        # Eval mode for the generator.
        self.generator.eval()

        img = self.generate(z)

        # Post process.
        img = self.post_process(img)

        return img

    @property
    def device(self):
        return next(self.parameters()).device




