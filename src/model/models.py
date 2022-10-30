import torch
from torch import nn
from torch import optim


class NeuralNetworkB0(nn.Module):
    def __init__(self):
        super(NeuralNetworkB0, self).__init__()

        # downloading pretrained efficient net model
        self.efficientnet = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_efficientnet_b0',
            pretrained=True)
        self.efficientnet.stem.conv = nn.Conv2d(1,
                                                32,
                                                kernel_size=(3, 3),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                bias=False)
        # adding classifier layer
        self.efficientnet.classifier.fc = nn.Linear(1280, 1, bias=True)

    def forward(self, x):
        return self.efficientnet(x)


############################################################################################
############################################################################################


class NeuralNetworkB4(nn.Module):
    def __init__(self):
        super(NeuralNetworkB4, self).__init__()
        # downloading pretrained efficient net model
        self.efficientnet = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_efficientnet_b4',
            pretrained=True)
        self.efficientnet.stem.conv = nn.Conv2d(1,
                                                48,
                                                kernel_size=(3, 3),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                bias=False)
        # adding classifier layer
        self.efficientnet.classifier.fc = nn.Linear(1792, 1, bias=True)

        # settings model for training parameters
        # efficient_net has four block {stem layers features classifier}
        for param in self.efficientnet.stem.parameters():
            param.requires_grad_(False)
        for param in self.efficientnet.layers.parameters():
            param.requires_grad_(False)
        for param in self.efficientnet.classifier.parameters():
            param.requires_grad_(True)
        for param in self.efficientnet.features.parameters():
            param.requires_grad_(True)

    def forward(self, x):
        return self.efficientnet(x)
