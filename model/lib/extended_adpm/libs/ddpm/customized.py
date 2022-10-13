import logging
import torch
from .model import Model, ResnetBlock, Normalize, get_timestep_embedding, nonlinearity


class Model4Pretrained(Model):
    def __init__(self, head_out_ch, mode="simple", **kwargs):
        super().__init__(**kwargs)
        self.requires_grad_(False)
        self.mode = mode
        logging.info('Model4Pretrained with mode={}'.format(self.mode))
        if mode == 'simple':
            self.before_out = lambda x, temb: x
        elif mode == 'complex':
            self.before_out = ResnetBlock(in_channels=self.block_in,
                                          out_channels=self.block_in,
                                          temb_channels=self.temb_ch,
                                          dropout=self.dropout)
        else:
            raise NotImplementedError

        self.norm_out2 = Normalize(self.block_in)
        self.conv_out2 = torch.nn.Conv2d(self.block_in,
                                         head_out_ch,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        out = self.conv_out(nonlinearity(self.norm_out(h)))
        out2 = self.conv_out2(nonlinearity(self.norm_out2(self.before_out(h, temb))))
        res = torch.cat([out, out2], dim=1)
        return res
