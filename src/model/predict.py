import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import optim
from train import train, test

import logging
# setting up logger
logger = logging.getLogger('Model')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('model_logs.log', mode="a")
fh.setLevel(logging.INFO)
logger.addHandler(fh)
# console output off
logger.propagate = False


# args to model
args = dict()
args['version'] = 'v0'
args['model'] = model
args['criterion'] = criterion
args['optimizer'] = optimizer
args['epochs'] = 20
args['plotting'] = False
args['perform_testing'] = True
