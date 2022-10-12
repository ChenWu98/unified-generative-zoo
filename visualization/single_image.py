# Created by Chen Henry Wu
import os
import math

from utils.file_utils import save_images
import torch.nn.functional as F


class Visualizer(object):

    def __init__(self, args):
        self.args = args

    def visualize(self,
                  images,
                  model,
                  description: str,
                  save_dir: str,
                  step: int,
                  ):
        if images is None:
            return

        # Just visualize the first 64 images.
        images = images[:64, :, :, :]

        save_images(
            images,
            output_dir=save_dir,
            file_prefix=description,
            nrows=int(math.sqrt(len(images))),
            iteration=step,
        )

        # Lower resolution
        images_256 = F.interpolate(
            images,
            (256, 256),
            mode='bicubic',
        )
        save_images(
            images_256,
            output_dir=save_dir,
            file_prefix=f'{description}_256',
            nrows=int(math.sqrt(len(images))),
            iteration=step,
        )


