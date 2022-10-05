import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.io import read_image


class PneumothoraxImgDataset(Dataset):
    def __init__(self, annotations_file, img_dir, dim=256):
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


Test_Dataset = PneumothoraxImgDataset('data/processed/test_data.csv',
                                      'data/external/small_train_data_set')
Val_Dataset = PneumothoraxImgDataset('data/processed/val_data.csv',
                                     'data/external/small_train_data_set')
Train_Dataset = PneumothoraxImgDataset('data/processed/train_data.csv',
                                       'data/external/small_train_data_set')


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

        train_loader = DataLoader(train_dataset, batch_size=64)
        val_loader = DataLoader(val_dataset, batch_size=32)

        print('-----------Trainning in Progress --------------')
        for idx, data in tqdm(enumerate(train_loader),
                              total=len(train_loader),
                              position=0,
                              leave=True):
            images, labels = data
            images = images.type(torch.float32).to(device)
            optimizer.zero_grad()

            outputs = model(images)
            labels = labels.type(torch.LongTensor).to(device)

            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_accuracy += torch.sum(preds == labels.data)

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
            labels = labels.type(torch.LongTensor).to(device)

            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            running_accuracy += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_dataset)
        val_accuracy = running_accuracy / len(val_dataset)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'\nVal Loss: {val_loss:.4f} Val Acc.: {val_accuracy:.4f}\n')

    return model, train_acc, train_losses, val_losses, val_accuracies


def test(model, criterion):
    test_loader = DataLoader(Test_Dataset, batch_size=32)

    model.eval()

    running_loss = 0
    running_accuracy = 0

    print('-------Testing Model------------')
    for idx, data in tqdm(enumerate(test_loader),
                          total=len(test_loader),
                          position=0,
                          leave=True):
        images, labels = data
        images = images.to(device)

        outputs = model(images)
        labels = labels.type(torch.LongTensor).to(device)

        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * images.size(0)
        running_accuracy += torch.sum(preds == labels.data)

    test_loss = running_loss / len(Test_Dataset)
    test_accuracy = running_accuracy / len(Test_Dataset)

    print(f'\nTest Loss: {test_loss:.5f} Test Acc.: {test_accuracy:.5f}\n')

    return test_loss, test_accuracy
