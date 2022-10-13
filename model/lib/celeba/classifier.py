# Created by Chen Henry Wu
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class ResNet50(nn.Module):
    def __init__(self, n_classes=1, pretrained=True, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Linear(2048, hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        outputs = self.fc(self.dropout(self.relu(features)))

        return outputs


class Classifier(nn.Module):
    def __init__(self, class_name):
        super(Classifier, self).__init__()

        # Face classifier preprocess.
        self.classifier_process = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        # Setup classifier.
        self.model = ResNet50(n_classes=1, pretrained=True)
        self.model.load_state_dict(torch.load('ckpts/celeba/{}/best.pth'.format(class_name), map_location='cpu')['model'])

    def forward(self, img):
        # Eval mode.
        self.model.eval()

        # Classifier process.
        img = self.classifier_process(img)

        outputs = self.model(img).squeeze(1)

        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
