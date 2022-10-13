from misc import str2bool, parse_sde, parse_schedule, create_sample_config, create_nll_config
import argparse
from interface.runner.runner import run_evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--phase', type=str, required=True)
    parser.add_argument('--description', type=str)
    parser.add_argument('--ckpt', type=str, default='best')

    hparams_types = dict(
        pretrained_path=str,  # path to evaluated model
        method=str,
        sample_steps=int,
        n_samples=int,
        batch_size=int,  # the total batch (over all devices)
        seed=int,
        # hyperparameters of architecture
        mode=str,
        # hyperparameters of DPMs with discrete timesteps
        schedule=str,
        rev_var_type=str,
        forward_type=str,
        eta=float,
        trajectory=str,
        clip_sigma_idx=int,
        clip_pixel=int,
        avg_cov=str2bool,
        ms_eps_path=str,
        # hyperparameters of DPMs with continuous timesteps (SDE)
        sde=str,
        reverse_type=str,
        t_init=float,
    )
    for hparam, typ in hparams_types.items():
        parser.add_argument(f'--{hparam}', type=typ)

    args = parser.parse_args()

    args.hparams = {key: getattr(args, key) for key in hparams_types.keys() if getattr(args, key) is not None}
    if 'schedule' in args.hparams:
        args.hparams['schedule'] = parse_schedule(args.hparams['schedule'])
    if 'sde' in args.hparams:
        args.hparams['sde'] = parse_sde(args.hparams['sde'])

    return args


def main():
    args = parse_args()

    if args.dataset == 'cifar10':
        from configs.cifar10 import get_evaluate_config
    elif args.dataset == 'celeba64':
        from configs.celeba64 import get_evaluate_config
    elif args.dataset == 'imagenet64':
        from configs.imagenet64 import get_evaluate_config
    elif args.dataset == 'lsun_bedroom':
        from configs.lsun import get_evaluate_config
    else:
        raise NotImplementedError

    keys = ['forward_type', 'eta', 'rev_var_type', 'reverse_type', 'sample_steps', 'trajectory', 'n_samples',
            'clip_sigma_idx', 'clip_pixel', 'avg_cov', 't_init', 'seed']

    if args.phase == 'sample4test':
        args.hparams.setdefault('n_samples', 50000)  # 5w samples for FID by default
        hparams = {**args.hparams, 'pretrained_path': args.pretrained_path, **args.hparams}
        config = create_sample_config(get_evaluate_config, args.workspace, args.ckpt, hparams, keys, args.description)
        run_evaluate(config)
    elif args.phase == "nll4test":
        args.hparams.setdefault('n_samples', None)  # all test samples
        hparams = {**args.hparams, 'pretrained_path': args.pretrained_path, **args.hparams}
        config = create_nll_config(get_evaluate_config, args.workspace, args.ckpt, hparams, keys, args.description)
        run_evaluate(config)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
