# Created by Chen Henry Wu
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import clip

from ..model_utils import requires_grad
from ..lib.diffaug.DiffAugment_pytorch import DiffAugment

POLICY = 'color,translation,resize,cutout'


class CLIPEnergy(nn.Module):
    def __init__(self, text, clip_models, clip_model_weights):
        super(CLIPEnergy, self).__init__()

        self.text = text
        self.replicate = 50

        self.clip_loss_models = nn.ModuleList(
            [
                CLIPEnergyComponent(clip_model=model_name)
                for model_name in clip_models
            ]
        )

        self.clip_model_weights = clip_model_weights

        # Freeze.
        requires_grad(self.clip_loss_models, False)

    @ staticmethod
    def prepare_inputs(**kwargs):
        return {
            'img': kwargs['img'],
        }

    def forward(self, img):
        # Eval mode for CLIP models.
        self.clip_loss_models.eval()

        B, C, H, W = img.shape

        # Diff augmentation.
        img_aug = torch.cat(
            [
                DiffAugment(_img.expand(self.replicate, -1, -1, -1), policy=POLICY)
                for _img in img
            ],
            dim=0,
        )  # (B * replicate, 3, H, W)

        clip_loss = None
        for model, weight in zip(self.clip_loss_models, self.clip_model_weights):
            loss = model(img_aug, self.text).view(B, self.replicate).mean(1)
            if clip_loss is None:
                clip_loss = loss * weight
            else:
                clip_loss += loss * weight

        return clip_loss


class CLIPEnergyComponent(nn.Module):
    def __init__(self,
                 clip_model,
                 ):
        super(CLIPEnergyComponent, self).__init__()

        self.model, clip_preprocess = clip.load(clip_model, device="cpu")  # cpu allows for fp32 loading.

        self.preprocess = transforms.Compose(  # Already un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            clip_preprocess.transforms[:2] +  # Skip ToRGB and ToTensor
            clip_preprocess.transforms[4:]
        )

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, class_str: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize([class_str]).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def global_clip_loss(self, img: torch.Tensor, text) -> torch.Tensor:

        image_encoding = self.get_image_features(img)  # Already normalized.
        text_encoding = self.get_text_features(text)  # Already normalized.

        return (
                1. - torch.einsum('bz,tz->bt', image_encoding, text_encoding)
        ).mean(1)

    def forward(self,
                target_img: torch.Tensor,
                target_class: str,
                ):

        return self.global_clip_loss(target_img, target_class)

    @property
    def device(self):
        return next(self.parameters()).device

