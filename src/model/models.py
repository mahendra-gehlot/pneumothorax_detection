import torch
from torch import nn
from torch import optim


class NeuralNetworkB0(nn.Module):
    def __init__(self):
        super(NeuralNetworkB0, self).__init__()
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
        self.efficientnet.classifier.fc = nn.Linear(1280,
                                                    1,
                                                    bias=True)
        for params in self.efficientnet.parameters():
            params.requires_grad = True

        self.drop_out = nn.Dropout(0.25)

    def forward(self, x):
        return self.drop_out(self.efficientnet(x))


############################################################################################
############################################################################################


class NeuralNetworkB4(nn.Module):
    def __init__(self):
        super(NeuralNetworkB4, self).__init__()
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
        self.efficientnet.classifier.fc = nn.Linear(1792,
                                                    1,
                                                    bias=True)
        for params in self.efficientnet.parameters():
            params.requires_grad = True
        self.drop_out = nn.Dropout(0.25)

    def forward(self, x):
        return self.drop_out(self.efficientnet(x))
