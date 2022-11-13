

def get_gan_wrapper(args, target=False):

    kwargs = {}
    for kw, arg in args:
        if kw != 'gan_type':
            if (not kw.startswith('source_')) and (not kw.startswith('target_')):
                kwargs[kw] = arg
            else:
                if target and kw.startswith('target_'):
                    final = kw[len('target_'):]
                    kwargs[f'source_{final}'] = arg
                elif (not target) and kw.startswith('source_'):
                    kwargs[kw] = arg

    if args.gan_type == "StyleGAN2":
        from .stylegan2_wrapper import StyleGAN2Wrapper
        return StyleGAN2Wrapper(**kwargs)
    elif args.gan_type == "StyleGAN-XL":
        from .styleganxl_wrapper import StyleGANXLWrapper
        return StyleGANXLWrapper(**kwargs)
    elif args.gan_type == "StyleSwin":
        from .styleswin_wrapper import StyleSwinWrapper
        return StyleSwinWrapper(**kwargs)
    elif args.gan_type == "StyleNeRF":
        from .stylenerf_wrapper import StyleNeRFWrapper
        return StyleNeRFWrapper(**kwargs)
    elif args.gan_type == "NVAETrunc":
        from .nvae_wrapper_trunc import NVAETruncWrapper
        return NVAETruncWrapper(**kwargs)
    elif args.gan_type == "DiffAE":
        from .diffae_wrapper import DiffAEWrapper
        return DiffAEWrapper(**kwargs)
    elif args.gan_type == "StyleSDF":
        from .stylesdf_wrapper import StyleSDFWrapper
        return StyleSDFWrapper(**kwargs)
    elif args.gan_type == "LatentDiff":
        from .latentdiff_wrapper import LatentDiffWrapper
        return LatentDiffWrapper(**kwargs)
    elif args.gan_type == "ExtendedAnalyticDPM":
        from .extended_adpm_wrapper import ExtendedAnalyticDPMWrapper
        return ExtendedAnalyticDPMWrapper(**kwargs)
    elif args.gan_type == "EG3D":
        from .eg3d_wrapper import EG3DWrapper
        return EG3DWrapper(**kwargs)
    elif args.gan_type == "DDGAN":
        from .ddgan_wrapper import DDGANWrapper
        return DDGANWrapper(**kwargs)
    elif args.gan_type == "GIRAFFE-HD":
        from .giraffehd_wrapper import GIRAFFEHDWrapper
        return GIRAFFEHDWrapper(**kwargs)
    elif args.gan_type == "DiffusionStyleGAN2":
        from .diffusion_stylegan2_wrapper import DiffusionStyleGAN2Wrapper
        return DiffusionStyleGAN2Wrapper(**kwargs)
    else:
        raise ValueError()

