import torch
import torch.nn as nn
import numpy as np

from .model_utils import requires_grad, MAX_SAMPLE_SIZE
from .gan_wrapper.get_gan_wrapper import get_gan_wrapper
from .energy.get_energy import get_energy, parse_key


class LangevinDynamics(nn.Module):

    def __init__(self, args):
        super(LangevinDynamics, self).__init__()
        self.args = args
        self.step_size = args.model.step_size
        self.n_steps = args.model.n_steps
        self.metric = args.model.metric
        if self.metric is not None:
            self.best_z = None
            self.best_metric = float('inf')

        # Set up gan_wrapper
        self.gan_wrapper = get_gan_wrapper(args.gan)
        requires_grad(self.gan_wrapper, True)  # Otherwise, no trainable params.

        # Energy
        self.energy_names, self.energy_weights, self.energy_modules = [], [], nn.ModuleList()
        for key, value in args:
            key, suffix = parse_key(key)
            if key.endswith('Energy') and ((suffix is None) or isinstance(suffix, int)):
                self.energy_names.append(key)
                self.energy_weights.append(value.weight)
                energy_kwargs = {kw: arg for kw, arg in value if kw != 'weight'}
                self.energy_modules.append(
                    get_energy(name=key, energy_kwargs=energy_kwargs, gan_wrapper=self.gan_wrapper)
                )
            elif key.endswith('Energy') and suffix == 'Pair':
                raise NotImplementedError()

        # Fixed noise for better visualization.
        self.register_buffer(
            "fixed_z",
            torch.randn(MAX_SAMPLE_SIZE, self.gan_wrapper.latent_dim),
        )

    def get_z_gaussian(self, sample_id=None):
        if self.training:
            bsz = sample_id.shape[0]
            z = torch.randn(bsz, self.gan_wrapper.latent_dim, device=self.device)
        else:
            assert sample_id.dim() == 1
            z = self.fixed_z[sample_id, :]

        return z

    @torch.enable_grad()
    def _langevin_dynamics_step(self, z, class_label):
        z = z.detach().requires_grad_()

        if getattr(self.gan_wrapper, "model_embedding_space", False):
            assert not getattr(self.gan_wrapper, "enforce_class_input", False)
            raise NotImplementedError()
        elif getattr(self.gan_wrapper, "enforce_class_input", False):
            assert not getattr(self.gan_wrapper, "model_embedding_space", False)
            assert class_label is not None
            img = self.gan_wrapper(z=z, class_label=class_label)
        else:
            img = self.gan_wrapper(z=z)

        losses = dict()
        weighted_loss = torch.zeros(z.shape[0], device=self.device).float()
        # Energy.
        for name, weight, module in zip(self.energy_names, self.energy_weights, self.energy_modules):
            inputs = module.prepare_inputs(img=img, z=z)
            loss = module(**inputs)
            losses[name] = loss
            weighted_loss += weight * loss
            print(name, loss)
        print('-' * 50)
        weighted_loss = weighted_loss.sum(0)

        # Gradient.
        grad = torch.autograd.grad(
            (weighted_loss, ),
            (z, ),
        )[0]
        assert grad.shape == z.shape

        # Update best z.
        if self.metric is not None:
            metric = losses[self.metric]
            if metric.item() < self.best_metric:
                self.best_z = z.detach().clone()
                self.best_metric = metric.item()

        # Noise.
        noise = torch.randn_like(grad) * np.sqrt(self.step_size)

        # Update.
        z = (z - self.step_size / 2 * grad + noise).detach()

        return z

    def langevin_dynamics(self, z, class_label):
        for step in range(self.n_steps):
            print(f'Step {step}')
            z = self._langevin_dynamics_step(z, class_label)
        return z

    def forward(self, sample_id, class_label=None):
        assert not self.training

        z = self.get_z_gaussian(sample_id=sample_id)  # (B, style_dim)

        # Init best z.
        if self.metric is not None:
            self.best_z = z
            self.best_metric = float('inf')

        if self.args.gan.gan_type in ["LatentDiff", "DiffAE"]:
            self.gan_wrapper.train()
        z = self.langevin_dynamics(z, class_label)
        self.gan_wrapper.eval()

        # Load best z.
        if self.metric is not None:
            z = self.best_z

        if getattr(self.gan_wrapper, "model_embedding_space", False):
            assert not getattr(self.gan_wrapper, "enforce_class_input", False)
            raise NotImplementedError()
        elif getattr(self.gan_wrapper, "enforce_class_input", False):
            assert not getattr(self.gan_wrapper, "model_embedding_space", False)
            assert class_label is not None
            img = self.gan_wrapper(z=z, class_label=class_label)
        else:
            img = self.gan_wrapper(z=z)

        losses = dict()
        weighted_loss = 0
        # Energy.
        for name, weight, module in zip(self.energy_names, self.energy_weights, self.energy_modules):
            inputs = module.prepare_inputs(img=img, z=z)
            loss = module(**inputs)
            losses[name] = loss
            weighted_loss += weight * loss
        print(losses)

        return img, weighted_loss, losses

    @property
    def device(self):
        return next(self.parameters()).device


Model = LangevinDynamics
