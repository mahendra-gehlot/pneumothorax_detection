import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import optim
from train_testing import train, test

import logging
# setting up logger
logger = logging.getLogger('Model')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('model_logs.log', mode="a")
fh.setLevel(logging.INFO)
logger.addHandler(fh)
# console output off
logger.propagate = False
####################################################################################


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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


device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_matric(train_acc, train_loss, val_loss, val_acc):
    """plotting accuracies and losses in iteration"""

    sns.set_theme()

    # creating data for x axes
    epochs = len(train_acc)
    epoch_ids = [epoch for epoch in range(1, epochs + 1)]

    # dividing in sub-plots
    plt.figure()
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(15, 8))

    # plot 1
    sns.lineplot(ax=axes[0], x=epoch_ids, y=train_acc)
    sns.lineplot(ax=axes[0], x=epoch_ids, y=val_acc)
    axes[0].set_title('Model Accuracy')
    axes[0].legend(['Train Acc', 'Val Acc'])

    # plot2
    sns.lineplot(ax=axes[1], x=epoch_ids, y=train_loss)
    sns.lineplot(ax=axes[1], x=epoch_ids, y=val_loss)
    axes[1].set_title('Model Losses')
    axes[1].legend(['Train Loss', 'Val Loss'])

    plt.show()

    return None


def execute(version,
            model,
            criterion,
            optimizer,
            epochs,
            plotting=True,
            perform_testing=True):
    logger.info(f'Version: {version}\n')
    trained_model, train_acc, train_loss, val_loss, val_acc = train(
        model, criterion, optimizer, num_of_epochs=epochs)

    logger.info('Epoch   Training Accuracy  Validation Accuracy')
    for idx in range(len(train_acc)):
        logger.info(
            f'{idx + 1}      {train_acc[idx]:.5f}              {val_acc[idx]:.5f}'
        )

    if plotting:
        print(type(train_acc[0]))
        plot_matric(train_acc, train_loss, val_loss, val_acc)

    if perform_testing:
        test_loss, test_acc = test(trained_model, criterion)
        logger.info(f'Testing Accuracy {test_acc:.5f}')

    return None


# model definition
model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# args to model
args = dict()
args['version'] = 'v0'
args['model'] = model
args['criterion'] = criterion
args['optimizer'] = optimizer
args['epochs'] = 20
args['plotting'] = False
args['perform_testing'] = True