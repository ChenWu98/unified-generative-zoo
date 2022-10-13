import configs.default as default
import interface.evaluators as evaluators
from .models import *
import datetime
from core.diffusion.schedule import NamedSchedule


def get_evaluate_config(**hparams):
    hparams.setdefault('method', 'pred_eps')
    hparams.setdefault('schedule', NamedSchedule('linear', 1000))
    hparams['N'] = hparams['schedule'].N

    config = ml_collections.ConfigDict()
    config.seed = hparams.get('seed', 1234)
    config.deterministic = hparams.get('deterministic', False)
    config.date = hparams.get('date', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    config.dataset = default.get_cifar10_config(**hparams)
    config.models = ml_collections.ConfigDict()

    if hparams['method'] == 'pred_eps':
        config.models.model = get_iddpm_unet_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps', **hparams)
    elif hparams['method'] == 'pred_x0':
        config.models.model = get_iddpm_unet_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='x0', **hparams)
    elif hparams['method'] == 'pred_eps_eps2':
        config.models.model = get_iddpm_unet_config(out_channels=6, **hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_eps2', **hparams)
    elif hparams['method'] == 'pred_eps_epsc':
        config.models.model = get_iddpm_unet_config(out_channels=6, **hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_epsc', **hparams)
    elif hparams['method'] == 'pred_eps_iddpm':
        config.models.model = get_iddpm_unet_config(out_channels=6, **hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_iddpm', **hparams)
    elif hparams['method'] == 'pred_eps_eps2_pretrained':
        config.models.model = get_iddpm_unet_double_pretrained_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_eps2', **hparams)
    elif hparams['method'] == 'pred_eps_epsc_pretrained':
        config.models.model = get_iddpm_unet_double_pretrained_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_epsc', **hparams)
    elif hparams['method'] == 'pred_eps_ct2dt':
        config.models.model = get_score_sde_ncsnpp_config(**hparams)
        config.wrapper = default.get_ct2dt_wrapper_config(typ='eps', **hparams)
    elif hparams['method'] == 'pred_eps_eps2_pretrained_ct2dt':
        config.models.model = get_score_sde_ncsnpp_double_pretrained_config(**hparams)
        config.wrapper = default.get_ct2dt_wrapper_config(typ='eps_eps2', **hparams)
    elif hparams['method'] == 'pred_eps_epsc_pretrained_ct2dt':
        config.models.model = get_score_sde_ncsnpp_double_pretrained_config(**hparams)
        config.wrapper = default.get_ct2dt_wrapper_config(typ='eps_epsc', **hparams)
    elif hparams['method'] == 'pred_eps_ct':
        config.models.model = get_score_sde_ncsnpp_config(**hparams)
        config.wrapper = default.get_ct_wrapper_config(typ='eps', **hparams)
    else:
        raise NotImplementedError

    config.evaluator = evaluator = ml_collections.ConfigDict()
    evaluator.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.options = options = ml_collections.ConfigDict()

    if hparams['method'] in ['pred_eps', 'pred_x0', 'pred_eps_eps2', 'pred_eps_epsc', 'pred_eps_iddpm',
                             'pred_eps_eps2_pretrained', 'pred_eps_epsc_pretrained',
                             'pred_eps_ct2dt', 'pred_eps_eps2_pretrained_ct2dt', 'pred_eps_epsc_pretrained_ct2dt']:
        evaluator.cls = evaluators.DTDPMEvaluator
        if hparams['task'] == 'sample2dir':
            options.sample2dir = default.get_sample2dir_config(**hparams)
        elif hparams['task'] == 'nll':
            options.nll = default.get_nll_config(**hparams)
        elif hparams['task'] == 'save_ms_eps':
            options.save_ms_eps = default.get_save_ms_eps_config(**hparams)
        elif hparams['task'] == 'save_nll_terms':
            options.save_nll_terms = default.get_save_nll_terms_config(**hparams)
        else:
            raise NotImplementedError
    elif hparams['method'] in ['pred_eps_ct']:
        evaluator.cls = evaluators.SDEEvaluator
        if hparams['task'] == 'sample2dir':
            options.sample2dir = default.get_sde_sample2dir_config(**hparams)
        elif hparams['task'] == 'nll':
            options.nll = default.get_sde_nll_config(**hparams)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return config
