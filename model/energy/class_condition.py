from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..lib.celeba.classifier import Classifier

from ..model_utils import requires_grad


class ClassEnergy(nn.Module):
    def __init__(self, classes, binaries, weights):
        super(ClassEnergy, self).__init__()

        self.classes = classes
        self.binaries = binaries
        self.weights = weights
        assert len(classes) == len(binaries) == len(weights)

        # Set up face classifier
        self.classifiers = nn.ModuleList()
        for i in range(len(classes)):
            self.classifiers.append(Classifier(classes[i]))
        # Freeze.
        requires_grad(self.classifiers, False)

    @ staticmethod
    def prepare_inputs(**kwargs):
        return {
            'img': kwargs['img'],
        }

    def forward(self, img):
        # Eval mode for the image classifier.
        self.classifiers.eval()

        # Classify.
        class_loss = []
        for i in range(len(self.classes)):
            prob = F.sigmoid(self.classifiers[i](img))
            if self.binaries[i] == 0:
                class_loss.append(
                    - torch.log(1 - prob + 1e-8) * self.weights[i]
                )
            elif self.binaries[i] == 1:
                class_loss.append(
                    - torch.log(prob + 1e-8) * self.weights[i]
                )
            else:
                raise ValueError('Invalid binary value.')

        # Losses
        class_loss = torch.stack(class_loss, dim=1).sum(dim=1)

        return class_loss

