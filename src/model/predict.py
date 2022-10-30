import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from train import PneumothoraxImgDataset

# looking device to run training
device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load('model/infer_model.pt')

Test_Dataset = PneumothoraxImgDataset('data/processed/test_data.csv',
                                      'data/external/small_train_data_set')
test_loader = DataLoader(Test_Dataset, batch_size=16)

model = model.to(device)

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
    images = images.to(device)
    outputs = model(images)
    labels = labels.type(torch.float32).to(device)
    pro_predict = torch.reshape(outputs, (-1, ))
    weights = torch.tensor([0.2 if x else 0.8 for x in labels]).to(device)
    criterion.weight = weights
    loss = criterion.forward(pro_predict, labels)
    running_loss += loss.item() * images.size(0)
    running_accuracy += torch.sum((pro_predict > 0.0) == labels.data)

    predicts.extend((pro_predict > 0.0).cpu())
    labels_all.extend(labels.cpu())

test_loss = running_loss / len(Test_Dataset)
test_accuracy = running_accuracy / len(Test_Dataset)

print(f'\nTest Loss: {test_loss:.5f} Test Acc.: {test_accuracy:.5f}\n')

reports = classification_report(np.array(labels_all), np.array(predicts))

print(f'Classification report: \n {reports}')
