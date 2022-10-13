
__all__ = ['set_seed', 'set_deterministic', 'backup_codes', 'backup_config']


import torch
import numpy as np
import os
import shutil
import pprint


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def set_deterministic(flag: bool):
    if flag:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def backup_codes(path, date):
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.realpath(os.path.join(current_path, os.pardir, os.pardir))

    path = os.path.join(path, f'codes_{date}')
    os.makedirs(path, exist_ok=True)

    names = ['core', 'interface', 'configs', 'libs', 'scripts', 'tools', 'useful']
    for name in names:
        if os.path.exists(os.path.join(root_path, name)):
            shutil.copytree(os.path.join(root_path, name), os.path.join(path, name))

    pyfiles = filter(lambda x: x.endswith('.py'), os.listdir(root_path))
    for pyfile in pyfiles:
        shutil.copy(os.path.join(root_path, pyfile), os.path.join(path, pyfile))


def backup_config(config, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f'config_{config.date}.txt')
    with open(path, 'w') as f:
        f.write(pprint.pformat(config.to_dict()))
