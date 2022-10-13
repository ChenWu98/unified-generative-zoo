from .unet import UNetModel, TimestepEmbedSequential, ResBlock
import logging
import torch as th
import torch.nn as nn
from .nn import (
    SiLU,
    conv_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class UNetModel3OutChannels(UNetModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, timesteps, y=None):
        out = super().forward(x, timesteps, y)
        if out.size(1) == 6:
            out = out.split(3, dim=1)[0]
        return out


class UNetModel4Pretrained(UNetModel):
    def __init__(self, head_out_channels, mode='simple', **kwargs):
        super().__init__(**kwargs)
        self.requires_grad_(False)
        self.mode = mode
        logging.info('UNetModel4Pretrained with mode={}'.format(self.mode))
        if mode == 'simple':
            self.out2 = nn.Sequential(
                normalization(self.out_ch),
                SiLU(),
                zero_module(conv_nd(kwargs["dims"], self.model_channels, head_out_channels, 3, padding=1)),
            )
        elif mode == 'complex':
            self.out2 = TimestepEmbedSequential(
                ResBlock(
                    self.out_ch,
                    self.time_embed_dim,
                    self.dropout,
                    dims=kwargs["dims"],
                    use_checkpoint=self.use_checkpoint,
                    use_scale_shift_norm=kwargs["use_scale_shift_norm"],
                ),
                normalization(self.out_ch),
                SiLU(),
                zero_module(conv_nd(kwargs["dims"], self.model_channels, head_out_channels, 3, padding=1)),
            )
        else:
            raise NotImplementedError

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        # h = x.type(self.inner_dtype)
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        if out.size(1) == 6:
            out = out.split(3, dim=1)[0]
        out2 = self.out2(h) if self.mode == 'simple' else self.out2(h, emb)
        res = th.cat([out, out2], dim=1)
        return res
