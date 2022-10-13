import ml_collections


def get_iddpm_unet_config(**hparams):  # the model from improved ddpm
    from libs.iddpm import UNetModel
    config = ml_collections.ConfigDict()
    config.cls = UNetModel
    config.pretrained_path = hparams.get('pretrained_path', None)
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.in_channels = 3
    kwargs.model_channels = hparams.get('model_channels', 128)
    kwargs.out_channels = hparams.get('out_channels', 3)
    kwargs.num_res_blocks = hparams.get('num_res_blocks', 3)
    kwargs.attention_resolutions = hparams.get('attention_resolutions', (32 // 16, 32 // 8))
    kwargs.dropout = hparams.get('dropout', 0.3)
    kwargs.channel_mult = hparams.get('channel_mult', (1, 2, 2, 2))
    kwargs.conv_resample = hparams.get('conv_resample', True)
    kwargs.dims = hparams.get('dims', 2)
    kwargs.num_classes = hparams.get('num_classes', None)
    kwargs.use_checkpoint = hparams.get('use_checkpoint', False)
    kwargs.num_heads = hparams.get('num_heads', 4)
    kwargs.num_heads_upsample = hparams.get('num_heads_upsample', -1)
    kwargs.use_scale_shift_norm = hparams.get('use_scale_shift_norm', True)
    return config


def get_ddpm_unet_config(**hparams):  # the model from ddpm
    from libs.ddpm import Model
    config = ml_collections.ConfigDict()
    config.cls = Model
    config.pretrained_path = hparams.get('pretrained_path', None)
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.ch = hparams.get('ch', 128)
    kwargs.out_ch = hparams.get('out_ch', 3)
    kwargs.ch_mult = hparams.get('ch_mult', (1, 2, 2, 2))
    kwargs.num_res_blocks = hparams.get('num_res_blocks', 2)
    kwargs.attn_resolutions = hparams.get('attn_resolutions', (16,))
    kwargs.dropout = hparams.get('dropout', 0.1)
    kwargs.in_channels = 3
    kwargs.resolution = 32
    return config


def get_iddpm_unet_double_pretrained_config(**hparams):
    assert 'pretrained_path' in hparams
    from libs.iddpm import UNetModel4Pretrained
    config = get_iddpm_unet_config(**hparams)
    config.cls = UNetModel4Pretrained
    config.kwargs.head_out_channels = 3
    config.kwargs.mode = hparams.get('mode', 'simple')
    return config


def get_score_sde_ncsnpp_config(**hparams):
    from libs.score_sde import get_nscnpp_model
    config = ml_collections.ConfigDict()
    config.cls = get_nscnpp_model
    config.pretrained_path = hparams.get('pretrained_path', None)
    config.kwargs = kwargs = ml_collections.ConfigDict()
    return config


def get_score_sde_ncsnpp_double_pretrained_config(**hparams):
    assert 'pretrained_path' in hparams
    from libs.score_sde import get_nscnpp_model
    config = ml_collections.ConfigDict()
    config.cls = get_nscnpp_model
    config.pretrained_path = hparams.get('pretrained_path', None)
    config.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.for_pretrained = True
    return config
