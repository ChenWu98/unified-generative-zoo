# Created by Chen Henry Wu


def get_energy(name, energy_kwargs, gan_wrapper):

    if name == "CLIPEnergy":
        from .clip_guide import CLIPEnergy
        return CLIPEnergy(**energy_kwargs)
    elif name == "PriorZEnergy":
        from .prior_z import PriorZEnergy
        return PriorZEnergy()
    elif name == "IDSingleEnergy":
        from .id_single import IDSingleEnergy
        return IDSingleEnergy(**energy_kwargs)
    elif name == "ClassEnergy":
        from .class_condition import ClassEnergy
        return ClassEnergy(**energy_kwargs)
    else:
        raise ValueError()


def parse_key(key):
    if key.endswith('1'):
        return key[:-1], 1
    elif key.endswith('2'):
        return key[:-1], 2
    elif key.endswith('Pair'):
        return key[:-len('Pair')], 'Pair'
    else:
        return key, None
