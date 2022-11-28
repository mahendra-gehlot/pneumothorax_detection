import torch
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from torchvision.io import read_image
from torchvision import datasets, transforms
from models import NeuralNetworkB4


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
    def __init__(self, annotations_file, img_dir, dim=380):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(3),
            transforms.Resize((dim, dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
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


# looking device to run training

device = 'cpu'

model = NeuralNetworkB4()

model.load_state_dict(torch.load('model/infer_model.pt', map_location = torch.device('cpu')))

Test_Dataset = PneumothoraxImgDataset('data/processed/test_data.csv',
                                      'data/external/small_train_data_set')
test_loader = DataLoader(Test_Dataset, batch_size=4)

running_loss = 0
running_accuracy = 0

predicts = []
labels_all = []

criterion = torch.nn.BCEWithLogitsLoss()

print('-------Testing Model------------')
for idx, data in tqdm(enumerate(test_loader),
                      total=len(test_loader),
                      position=0,
                      leave=True):
    images, labels = data
    images = images.type(torch.float32).to(device)

    outputs = model(images)
    labels = labels.type(torch.float32).to(device)

    pro_predict = torch.reshape(outputs, (-1,))

    loss = criterion(pro_predict, labels)
    running_loss += loss.item() * images.size(0)
    running_accuracy += torch.sum((pro_predict > 0.0) == labels.data)

    predicts.extend((pro_predict > 0.0).cpu())
    labels_all.extend(labels.cpu())

test_loss = running_loss / len(Test_Dataset)
test_accuracy = running_accuracy / len(Test_Dataset)

print(f'\nTest Loss: {test_loss:.5f} Test Acc.: {test_accuracy:.5f}\n')

reports = classification_report(np.array(labels_all), np.array(predicts))

print(f'Classification report: \n {reports}')
