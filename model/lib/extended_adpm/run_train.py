from misc import parse_sde, parse_schedule, str2bool
import argparse
from interface.runner.runner import run_train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)

    hparams_types = dict(
        pretrained_path=str,  # use pretrained state
        method=str,
        seed=int,
        n_its=int,
        n_ckpts=int,
        random_flip=str2bool,
        lr=float,
        warmup_its=int,
        grad_clip=float,
        # hyperparameters of architecture
        mode=str,
        # hyperparameters of DPMs with discrete timesteps
        schedule=str,
        # hyperparameters of DPMs with continuous timesteps (SDE)
        sde=str,
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
        from configs.cifar10 import get_train_config
    elif args.dataset == 'celeba64':
        from configs.celeba64 import get_train_config
    elif args.dataset == 'imagenet64':
        from configs.imagenet64 import get_train_config
    elif args.dataset == 'lsun_bedroom':
        from configs.lsun import get_train_config
    else:
        raise NotImplementedError

    hparams = dict(workspace=args.workspace, **args.hparams)
    config = get_train_config(**hparams)
    run_train(config)


if __name__ == '__main__':
    main()
