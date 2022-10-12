import torch
from torch import nn
from torch import optim


class NeuralNetwork_B0(nn.Module):
    def __init__(self):
        super(NeuralNetwork_B0, self).__init__()
        self.classes = 2
        self.efficientnet = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_efficientnet_b0',
            pretrained=False)
        self.efficientnet.stem.conv = nn.Conv2d(1,
                                                32,
                                                kernel_size=(3, 3),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                bias=False)
        self.efficientnet.classifier.fc = nn.Linear(1280,
                                                    self.classes,
                                                    bias=True)

    def forward(self, x):
        return self.efficientnet(x)


############################################################################################
############################################################################################


class NeuralNetwork_B4(nn.Module):
    def __init__(self):
        super(NeuralNetwork_B4, self).__init__()
        self.classes = 2
        self.efficientnet = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_efficientnet_b4',
            pretrained=False)
        self.efficientnet.stem.conv = nn.Conv2d(1,
                                                48,
                                                kernel_size=(3, 3),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                bias=False)
        self.efficientnet.classifier.fc = nn.Linear(1792,
                                                    self.classes,
                                                    bias=True)

    def forward(self, x):
        return self.efficientnet(x)
