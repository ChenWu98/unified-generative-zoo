import configs.default as default
import interface.evaluators as evaluators
from .models import *
import datetime
from core.diffusion.schedule import NamedSchedule


def get_evaluate_config(**hparams):
    hparams.setdefault('method', 'pred_eps')
    hparams.setdefault('schedule', NamedSchedule('linear', 1000))
    hparams.setdefault('shift1', True)  # follow original DDPM
    hparams['N'] = hparams['schedule'].N

    config = ml_collections.ConfigDict()
    config.seed = hparams.get('seed', 1234)
    config.deterministic = hparams.get('deterministic', False)
    config.date = hparams.get('date', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    config.dataset = default.get_lsun_bedroom_config(**hparams)
    config.models = ml_collections.ConfigDict()
    if hparams['method'] == 'pred_eps':
        config.models.model = get_ddpm_unet_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps', **hparams)
    elif hparams['method'] == 'pred_eps_eps2_pretrained':
        config.models.model = get_ddpm_unet_double_pretrained_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_eps2', **hparams)
    elif hparams['method'] == 'pred_eps_epsc_pretrained':
        config.models.model = get_ddpm_unet_double_pretrained_config(**hparams)
        config.wrapper = default.get_dt_wrapper_config(typ='eps_epsc', **hparams)
    else:
        raise NotImplementedError

    config.evaluator = evaluator = ml_collections.ConfigDict()
    evaluator.cls = evaluators.DTDPMEvaluator
    evaluator.kwargs = kwargs = ml_collections.ConfigDict()
    kwargs.options = options = ml_collections.ConfigDict()

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

    return config
