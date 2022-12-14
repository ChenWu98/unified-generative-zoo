import torch
import argparse
import logging
import os
from PIL import Image
import json
from tqdm import tqdm

from utils.config_utils import get_config
from model.gan_wrapper.get_gan_wrapper import get_gan_wrapper

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1000,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = get_config(os.path.join('generate', f'{args.model}.cfg'))
    torch.cuda.manual_seed(0)
    generator = get_gan_wrapper(config.model).cuda()
    generator.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

    style_vectors = dict()
    for i in tqdm(range(args.num_images)):
        image, style = generator.sample_image_style()
        image = image[0]
        style = style[0]

        image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        image = Image.fromarray(image)
        file_name = f"image_{str(i).zfill(6)}.png"
        image.save(os.path.join(args.output_dir, 'images', file_name))
        style_vectors[file_name] = style.to("cpu")
    torch.save(style_vectors, os.path.join(args.output_dir, "style_vectors.pt"))


if __name__ == "__main__":
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)

    main()

