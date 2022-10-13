import ml_collections


def get_ddpm_unet_config(**hparams):  # the model from ddpm
    from libs.ddpm import Model
    config = ml_collections.ConfigDict()
    config.cls = Model
    config.pretrained_path = hparams.get('pretrained_path', None)
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.ch = hparams.get('ch', 128)
    kwargs.out_ch = hparams.get('out_ch', 3)
    kwargs.ch_mult = hparams.get('ch_mult', (1, 2, 2, 2, 4))
    kwargs.num_res_blocks = hparams.get('num_res_blocks', 2)
    kwargs.attn_resolutions = hparams.get('attn_resolutions', (16,))
    kwargs.dropout = hparams.get('dropout', 0.1)
    kwargs.in_channels = 3
    kwargs.resolution = 64
    return config


def get_ddpm_unet_double_pretrained_config(**hparams):
    assert 'pretrained_path' in hparams
    from libs.ddpm import Model4Pretrained
    config = get_ddpm_unet_config(**hparams)
    config.cls = Model4Pretrained
    config.kwargs.head_out_ch = 3
    config.kwargs.mode = hparams.get('mode', 'simple')
    return config
