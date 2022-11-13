import torch
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize

from ..model_utils import requires_grad
from ..lib.id_recognition.model_irse import Backbone

RESOLUTION = 112  # Resolution depends on the center crop AND 256 below.


class IDSingleEnergy(nn.Module):
    def __init__(self, ir_se50_weights, ref_paths):
        super(IDSingleEnergy, self).__init__()

        self.id_loss = IDSingleLoss(ir_se50_weights)
        self.ref_transform = Compose(
            [
                Resize(RESOLUTION),
                ToTensor(),
            ]
        )
        self.ref_paths = ref_paths

        # Freeze.
        requires_grad(self.id_loss, False)

    @ staticmethod
    def prepare_inputs(**kwargs):
        return {
            'img': kwargs['img'],
        }

    def forward(self, img):
        # Eval mode for the IDLoss module.
        self.id_loss.eval()

        references = torch.stack(
            [self.ref_transform(Image.open(path)) for path in self.ref_paths], dim=0
        ).to(img.device)
        disc_loss = self.id_loss(img, references)

        return disc_loss


class IDSingleLoss(nn.Module):
    """This loss follows StyleCLIP. """
    def __init__(self, ir_se50_weights):
        super(IDSingleLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=RESOLUTION, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(ir_se50_weights))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((RESOLUTION, RESOLUTION))

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, y):
        # Eval mode.
        self.facenet.eval()

        x_feats, y_feats = self.extract_feats(x), self.extract_feats(y)  # Features are already l2 normalized.
        diff_target = torch.einsum('bh,rh->br', x_feats, y_feats).mean(dim=1)
        loss = 1 - diff_target

        return loss
