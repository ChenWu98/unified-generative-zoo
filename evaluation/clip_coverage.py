# Created by Chen Henry Wu
import torch
from tqdm import tqdm
import torch.nn.functional as F

import lpips


class Evaluator(object):

    def __init__(self, args, meta_args):
        self.args = args
        self.meta_args = meta_args

        self.lpips_loss = lpips.LPIPS(net='vgg').cuda()

    def evaluate(self, images, model, weighted_loss, losses, data, split):
        """

        Args:
            images: list of images, or list of tuples of images
            model: model to evaluate
            weighted_loss: list of scalar tensors
            losses: dictionary of lists of scalar tensors
            data: list of dictionary
            split: str

        Returns:

        """
        # Eval mode for the LPIPS loss module.
        self.lpips_loss.eval()
        model = None

        assert split in ['eval', 'test']

        # Filter out NaN.
        images = [image for image, _CLIPEnergy in zip(images, losses['CLIPEnergy']) if not _CLIPEnergy.isnan()]
        CLIPEnergy = [_CLIPEnergy.item() for _CLIPEnergy in losses['CLIPEnergy'] if not _CLIPEnergy.isnan()]
        CLIPEnergy = torch.FloatTensor(CLIPEnergy).mean(0).item()
        PriorZEnergy = [_PriorZEnergy.item() for _PriorZEnergy, _CLIPEnergy in zip(losses['PriorZEnergy'], losses['CLIPEnergy']) if not _CLIPEnergy.isnan()]
        PriorZEnergy = torch.FloatTensor(PriorZEnergy).mean(0).item()

        # Add metrics here.
        # LPIPS metric.
        image1, image2 = [], []
        n = len(images)
        for i in range(n - 1):
            for j in range(i + 1, n):
                image1.append(images[i])
                image2.append(images[j])

        batch_size = 1
        with torch.no_grad():
            lpips_values_256 = []
            for b in tqdm(range(len(image1) // batch_size)):
                lpips_values_256.append(
                    self.lpips_loss(
                        F.interpolate(
                            torch.stack(image1[b * batch_size:(b + 1) * batch_size], dim=0),
                            (256, 256),
                            mode='bicubic',
                        ).cuda(),
                        F.interpolate(
                            torch.stack(image2[b * batch_size:(b + 1) * batch_size], dim=0),
                            (256, 256),
                            mode='bicubic',
                        ).cuda(),
                        normalize=True,
                    ).squeeze(3).squeeze(2).squeeze(1).cpu()
                )
            lpips_value_256 = torch.cat(lpips_values_256, dim=0).mean(0).item()

        summary = {
            "lpips_256": lpips_value_256,
            "CLIPEnergy_nan_removed": CLIPEnergy,
            "PriorZEnergy_nan_removed": PriorZEnergy,
            "kept_num": len(images),
        }

        return summary
