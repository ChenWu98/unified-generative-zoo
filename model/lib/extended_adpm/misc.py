import os
from typing import List
import ml_collections
import argparse
from core.diffusion.schedule import NamedSchedule
from core.diffusion.sde import VPSDE


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_schedule(schedule):
    if schedule.startswith('linear') or schedule.startswith('cosine'):
        typ, N = schedule.split('_')
        N = int(N)
        return NamedSchedule(typ, N)
    elif schedule.startswith('vpsde'):
        typ, N = schedule.split('_')
        N = int(N)
        return VPSDE().get_schedule(N)
    else:
        raise NotImplementedError


def parse_sde(sde):
    if sde == 'vpsde':
        return VPSDE()
    else:
        raise NotImplementedError


def sub_dict(dct: dict, *keys):
    return {key: dct[key] for key in keys if key in dct}


def dict2str(dct):
    pairs = []
    for key, val in dct.items():
        pairs.append("{}_{}".format(key, val))
    return "_".join(pairs)
