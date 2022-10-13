from core.diffusion.schedule import NamedSchedule
import configs.default as default
from .models import *


def get_train_config(**hparams):
    hparams.setdefault('method', 'pred_eps')
    hparams.setdefault('schedule', NamedSchedule('linear', 1000))
    hparams['N'] = hparams['schedule'].N

    config = default.get_train_config(**hparams)
    config.models = ml_collections.ConfigDict()
    config.dataset = default.get_cifar10_config(**hparams)
    if hparams['method'] == 'pred_eps':
        config.models.model = get_iddpm_unet_config(**hparams)
        config.criterion = default.get_dt_dsm_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps', **hparams)
    elif hparams['method'] == 'pred_x0':
        config.models.model = get_iddpm_unet_config(**hparams)
        config.criterion = default.get_dt_dsm0_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='x0', **hparams)
    elif hparams['method'] == 'pred_eps_eps2':
        hparams['rev_var_type'] = 'optimal'
        config.models.model = get_iddpm_unet_config(out_channels=6, **hparams)
        config.criterion = default.get_dt_dsdm_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_eps2', **hparams)
    elif hparams['method'] == 'pred_eps_epsc':
        hparams['rev_var_type'] = 'optimal'
        config.models.model = get_iddpm_unet_config(out_channels=6, **hparams)
        config.criterion = default.get_dt_dsdm_err_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_epsc', **hparams)
    elif hparams['method'] == 'pred_eps_iddpm':
        hparams['rev_var_type'] = 'optimal'
        config.models.model = get_iddpm_unet_config(out_channels=6, **hparams)
        config.criterion = default.get_lhybrid_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_iddpm', **hparams)
    elif hparams['method'] == 'pred_eps_eps2_pretrained':
        hparams['rev_var_type'] = 'optimal'
        config.models.model = get_iddpm_unet_double_pretrained_config(**hparams)
        config.criterion = default.get_dt_dsdm_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_eps2', **hparams)
    elif hparams['method'] == 'pred_eps_epsc_pretrained':
        hparams['rev_var_type'] = 'optimal'
        config.models.model = get_iddpm_unet_double_pretrained_config(**hparams)
        config.criterion = default.get_dt_dsdm_err_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_epsc', **hparams)
    elif hparams['method'] == 'pred_eps_eps2_pretrained_ct':
        config.models.model = get_score_sde_ncsnpp_double_pretrained_config(**hparams)
        config.criterion = default.get_ct_dsdm_config(**hparams)
        config.wrapper = default.get_ct_wrapper_config(typ='eps_eps2', **hparams)
    elif hparams['method'] == 'pred_eps_epsc_pretrained_ct':
        config.models.model = get_score_sde_ncsnpp_double_pretrained_config(**hparams)
        config.criterion = default.get_ct_dsdm_err_config(**hparams)
        config.wrapper = default.get_ct_wrapper_config(typ='eps_epsc', **hparams)
    elif hparams['method'] == 'pred_eps_ct':
        config.models.model = get_score_sde_ncsnpp_config(**hparams)
        config.criterion = default.get_ct_dsm_config(**hparams)
        config.wrapper = default.get_ct_wrapper_config(typ='eps', **hparams)
    else:
        raise NotImplementedError

    if hparams['method'] in ['pred_eps', 'pred_x0', 'pred_eps_eps2', 'pred_eps_epsc', 'pred_eps_iddpm',
                             'pred_eps_eps2_pretrained', 'pred_eps_epsc_pretrained',
                             'pred_eps_ct2dt', 'pred_eps_eps2_pretrained_ct2dt', 'pred_eps_epsc_pretrained_ct2dt']:
        config.evaluator = default.get_train_evaluator_config(**hparams)
    elif hparams['method'] in ['pred_eps_ct']:
        config.evaluator = default.get_sde_train_evaluator_config(**hparams)
    elif hparams['method'] in ['pred_eps_eps2_pretrained_ct', 'pred_eps_epsc_pretrained_ct']:
        pass
    else:
        raise NotImplementedError

    return config
