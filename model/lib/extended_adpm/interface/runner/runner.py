
__all__ = ["run_train", "run_evaluate", "run_timing_profile"]


from interface.runner.fit import naive_fit
from interface.runner.timing import timing
from interface.utils import set_seed, set_deterministic, backup_codes, backup_config
from interface.utils import ckpt, config_utils
from core.utils import global_device
from core.utils.managers import ModelsManager
import ml_collections


def merge_models(dest: ModelsManager, src: ModelsManager):
    if src is None:
        return dest
    _dict = {key: dest.get(key) for key in dest.keys()}
    for key, val in src.items():
        _dict[key] = val
    return ModelsManager(**_dict)


def run_train(config):
    config = ml_collections.FrozenConfigDict(config)
    set_seed(config.seed)
    set_deterministic(config.deterministic)
    backup_codes(config.backup_root, config.date)
    backup_config(config, config.backup_root)

    interact = config_utils.create_interact(config.interact)
    interact.report_machine()

    models = config_utils.create_models(config.models)
    ema_models, ema_rate = config_utils.create_ema(config, models)
    wrapper = config_utils.create_wrapper(config.wrapper, models)
    optimizers = config_utils.create_optimizers(config.optimizers, models)
    lr_schedulers = config_utils.create_lr_schedulers(config.lr_schedulers, optimizers)
    criterion = config_utils.create_criterion(config.criterion, wrapper, models, optimizers, lr_schedulers)

    dataset = config_utils.create_dataset(config.dataset)
    if config.dataset.use_val:
        train_dataset = dataset.get_train_data()
        val_dataset = dataset.get_val_data()
    else:
        train_dataset = dataset.get_train_val_data()
        val_dataset = None

    evaluator = None
    if 'evaluator' in config:
        merged_models = merge_models(models, ema_models)
        merged_wrapper = config_utils.create_wrapper(config.wrapper, merged_models)
        evaluator = config_utils.create_evaluator(config.evaluator, merged_wrapper, dataset, interact)

    naive_fit(criterion=criterion,
              train_dataset=train_dataset,
              batch_size=config.training.batch_size,
              n_its=config.training.n_its,
              n_ckpts=config.training.n_ckpts,
              ckpt_root=config.ckpt_root,
              interact=interact,
              evaluator=evaluator,
              val_dataset=val_dataset,
              val_fn=config_utils.create_val_fn(config, criterion),
              ckpt=ckpt.get_ckpt_by_it(config.ckpt_root),
              ema_models=ema_models,
              ema_rate=ema_rate
              )


def run_evaluate(config):
    config = ml_collections.FrozenConfigDict(config)
    set_seed(config.seed)
    set_deterministic(config.deterministic)
    backup_codes(config.backup_root, config.date)
    backup_config(config, config.backup_root)
    interact = config_utils.create_interact(config.interact)
    interact.report_machine()
    models = config_utils.create_models(config.models)  # use pretrained_path to load models
    wrapper = config_utils.create_wrapper(config.wrapper, models)
    dataset = config_utils.create_dataset(config.dataset)
    evaluator = config_utils.create_evaluator(config.evaluator, wrapper, dataset, interact)
    models.to(global_device())
    models.eval()
    evaluator.evaluate()


def run_timing_profile(config):
    config = ml_collections.FrozenConfigDict(config)
    set_seed(config.seed)
    set_deterministic(config.deterministic)
    backup_codes(config.backup_root, config.date)
    backup_config(config, config.backup_root)
    interact = config_utils.create_interact(config.interact)
    interact.report_machine()
    models = config_utils.create_models(config.models)
    wrapper = config_utils.create_wrapper(config.wrapper, models)
    optimizers = config_utils.create_optimizers(config.optimizers, models)
    lr_schedulers = config_utils.create_lr_schedulers(config.lr_schedulers, optimizers)
    criterion = config_utils.create_criterion(config.criterion, models, wrapper, optimizers, lr_schedulers)

    dataset = config_utils.create_dataset(config.dataset)
    if config.dataset.use_val:
        train_dataset = dataset.get_train_data()
    else:
        train_dataset = dataset.get_train_val_data()

    timing(criterion=criterion,
           train_dataset=train_dataset,
           batch_size=config.training.batch_size,
           n_its=config.training.n_its
           )
