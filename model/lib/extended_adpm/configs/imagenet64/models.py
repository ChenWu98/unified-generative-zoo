import ml_collections


def get_iddpm_unet_config(**hparams):  # the model from improved ddpm
    from libs.iddpm import UNetModel
    config = ml_collections.ConfigDict()
    config.cls = UNetModel
    config.pretrained_path = hparams.get('pretrained_path', None)
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.in_channels = 3
    kwargs.model_channels = hparams.get('model_channels', 128)
    kwargs.out_channels = hparams.get('out_channels', 6)
    kwargs.num_res_blocks = hparams.get('num_res_blocks', 3)
    kwargs.attention_resolutions = hparams.get('attention_resolutions', (64 // 16, 64 // 8))
    kwargs.dropout = hparams.get('dropout', 0.0)
    kwargs.channel_mult = hparams.get('channel_mult', (1, 2, 3, 4))
    kwargs.conv_resample = hparams.get('conv_resample', True)
    kwargs.dims = hparams.get('dims', 2)
    kwargs.num_classes = hparams.get('num_classes', None)
    kwargs.use_checkpoint = hparams.get('use_checkpoint', False)
    kwargs.num_heads = hparams.get('num_heads', 4)
    kwargs.num_heads_upsample = hparams.get('num_heads_upsample', -1)
    kwargs.use_scale_shift_norm = hparams.get('use_scale_shift_norm', True)
    return config


def get_iddpm_unet_out3_config(**hparams):
    from libs.iddpm import UNetModel3OutChannels
    config = get_iddpm_unet_config(**hparams)
    config.cls = UNetModel3OutChannels
    return config


def get_iddpm_unet_double_pretrained_config(**hparams):
    from libs.iddpm import UNetModel4Pretrained
    assert 'pretrained_path' in hparams
    config = get_iddpm_unet_config(**hparams)
    config.cls = UNetModel4Pretrained
    config.kwargs.head_out_channels = 3
    config.kwargs.mode = hparams['mode']
    return config
