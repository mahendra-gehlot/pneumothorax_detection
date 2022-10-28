import os
import copy
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from models import *

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.io import read_image

# setting up logger
import logging

logger = logging.getLogger('Model')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('model_logs.log', mode="a")
fh.setLevel(logging.INFO)
logger.addHandler(fh)

# console output off
logger.propagate = False


class PneumothoraxImgDataset(Dataset):
    """
        Custom Dataset for Pneumothorax Detection Dataset

        ...

        Attributes
        ----------
        annotations_file : str
            path of file dataset
        img_dir : str
            directory of images
    """
    def __init__(self, annotations_file, img_dir, dim=192):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((dim, dim)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label


# Custom dataset instantiate
Test_Dataset = PneumothoraxImgDataset('data/processed/test_data.csv',
                                      'data/external/small_train_data_set')
Val_Dataset = PneumothoraxImgDataset('data/processed/val_data.csv',
                                     'data/external/small_train_data_set')
Train_Dataset = PneumothoraxImgDataset('data/processed/train_data.csv',
                                       'data/external/small_train_data_set')

# looking device to run training
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, criterion, optimizer, num_of_epochs):
    train_losses = []
    train_acc = []
    val_losses = []
    val_accuracies = []

    for _, epoch in tqdm(enumerate(range(num_of_epochs))):
        print(f'\nEpoch {epoch + 1}/{num_of_epochs}')

        model.train()

        running_loss = 0.
        running_accuracy = 0.

        train_dataset, val_dataset = Train_Dataset, Test_Dataset

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

        print('-----------Training in Progress --------------')
        for idx, data in tqdm(enumerate(train_loader),
                              total=len(train_loader),
                              position=0,
                              leave=True):
            images, labels = data
            images = images.type(torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            sigmoid = nn.Sigmoid()
            labels = labels.type(torch.float32).to(device)
            pro_predict = sigmoid(torch.reshape(outputs, (-1,)))
            loss = criterion(pro_predict, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            running_accuracy += torch.sum(pro_predict == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = running_accuracy / len(train_dataset)

        train_losses.append(epoch_loss)
        train_acc.append(epoch_accuracy)

        print(
            f'Training Loss: {epoch_loss:.6f} Training Acc.: {epoch_accuracy:.6f}'
        )

        model.eval()

        running_loss = 0
        running_accuracy = 0

        print('-----------Validation in Progress --------------')

        for idx, data in tqdm(enumerate(val_loader),
                              total=len(val_loader),
                              position=0,
                              leave=True):
            images, labels = data
            images = images.type(torch.float32).to(device)

            outputs = model(images)
            sigmoid = nn.Sigmoid()
            labels = labels.type(torch.float32).to(device)

            pro_predict = sigmoid(torch.reshape(outputs, (-1,)))
            loss = criterion(pro_predict, labels)

            running_loss += loss.item() * images.size(0)
            running_accuracy += torch.sum(pro_predict == labels.data)

        val_loss = running_loss / len(val_dataset)
        val_accuracy = running_accuracy / len(val_dataset)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'\nVal Loss: {val_loss:.4f} Val Acc.: {val_accuracy:.4f}\n')

        return

    return model, train_acc, train_losses, val_losses, val_accuracies


def test(model, criterion):
    from sklearn.metrics import classification_report
    test_loader = DataLoader(Test_Dataset, batch_size=32)

    model.eval()

    running_loss = 0
    running_accuracy = 0

    predicts = []
    labels_all = []

    print('-------Testing Model------------')
    for idx, data in tqdm(enumerate(test_loader),
                          total=len(test_loader),
                          position=0,
                          leave=True):
        images, labels = data
        images = images.to(device)
        sigmoid = nn.Sigmoid()
        outputs = model(images)
        labels = labels.type(torch.float32).to(device)
        pro_predict = sigmoid(torch.reshape(outputs, (-1,)))
        loss = criterion(pro_predict, labels)

        running_loss += loss.item() * images.size(0)
        running_accuracy += torch.sum(pro_predict == labels.data)

        predicts.extend(pro_predict.cpu())
        labels_all.extend(labels.cpu())

    test_loss = running_loss / len(Test_Dataset)
    test_accuracy = running_accuracy / len(Test_Dataset)

    print(f'\nTest Loss: {test_loss:.5f} Test Acc.: {test_accuracy:.5f}\n')

    reports = classification_report(np.array(labels_all), np.array(predicts))

    print(f'Classification report: \n {reports}')

    return test_loss, test_accuracy, reports


def plot_losses_acc(version, train_acc, train_loss, val_loss, val_acc):
    """plotting accuracies and losses in iterations"""

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

    fig.savefig(f"reports/figures/{version}_acc_loss" + ".png", format='png')

    return None


def execute(version,
            model,
            criterion,
            optimizer,
            epochs,
            save_model,
            plotting=True,
            perform_testing=True):
    logger.info(f'Version: {version}\n')
    trained_model, train_acc, train_loss, val_loss, val_acc = train(
        model, criterion, optimizer, num_of_epochs=epochs)

    if save_model:
        torch.save(trained_model, f'model/infer_model.pt')

    logger.info('Epoch   Training Accuracy  Validation Accuracy')
    for idx in range(len(train_acc)):
        logger.info(
            f'{idx + 1}      {train_acc[idx]:.5f}              {val_acc[idx]:.5f}'
        )

    if plotting:
        def convert_to_cpu(gpu_data):
            cpu_data = []
            for unit in gpu_data:
                if isinstance(unit, float):
                    cpu_data.append(unit)
                else:
                    cpu_data.append(float(unit.cpu()))

            return cpu_data

        train_acc_cpu = convert_to_cpu(train_acc)
        train_loss_cpu = convert_to_cpu(train_loss)
        val_loss_cpu = convert_to_cpu(val_loss)
        val_acc_cpu = convert_to_cpu(val_acc)
        plot_losses_acc(version, train_acc_cpu, train_loss_cpu, val_loss_cpu, val_acc_cpu)

    if perform_testing:
        test_loss, test_acc, reports = test(trained_model, criterion)
        logger.info(f'Testing Accuracy {test_acc:.5f}')
        logger.info(f'Report: {reports}')

    return None


def run():
    parser = argparse.ArgumentParser(
        description='specify arguments for training')
    parser.add_argument('-m',
                        '--model',
                        choices=['B0', 'B4'],
                        default='B0',
                        help='select models of efficient-net')
    parser.add_argument('-v',
                        '--version',
                        required=True,
                        type=str,
                        help='specify version e.g v0 v1 v2...')
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=20,
                        help='Number of epochs for training')

    parser.add_argument('save_model',
                        type=bool,
                        default=False,
                        help='saves trained model')

    parser.add_argument('make_plots',
                        type=bool,
                        default=False,
                        help='plots of losses and accuracy are stored')

    args = parser.parse_args()

    if args.model == 'B0':
        logger.info(f'Model {args.model}')
        current_model = NeuralNetworkB0().to(device)
    elif args.model == 'B4':
        logger.info(f'Model {args.model}')
        current_model = NeuralNetworkB4().to(device)
    else:
        logger.info(f'Model {args.model}')
        current_model = NeuralNetworkB0().to(device)

    # adding weights to handle class imbalance
    weights = torch.tensor([(1597./2027.), (430./2027.)]).to(device)
    loss_criterion = nn.BCELoss()
    model_optimizer = optim.Adam(current_model.parameters(), lr=0.001)

    execute(args.version,
            current_model,
            loss_criterion,
            model_optimizer,
            args.epochs,
            save_model=args.save_model,
            plotting=args.make_plots,
            perform_testing=True)

    return None


if __name__ == '__main__':
    run()
