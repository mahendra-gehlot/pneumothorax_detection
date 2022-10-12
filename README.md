# Pneumothorax Binary Classification task

### Project Tree
```

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from https://www.kaggle.com/datasets/volodymyrgavrysh/pneumothorax-binary-classification-task
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    |
    ├── src                <- Source code for use in this project.
    │   ├── data           <- Scripts to download or generate data
    │   │   └── prepare_data.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── models.py  <- Contain py-torch models
    │   │   └── train.py   <- script for training and testing
    │   │   └── predict.py <- prediction for given image
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations

```

### Context
A pneumothorax occurs when air leaks into the space between your lung and chest wall. 
This air pushes on the outside of your lung and makes it collapse.  

### Implementation Details
The Effiecient-Net B0, B4 model are implemented for classification task.

### Dataset
```
https://www.kaggle.com/datasets/volodymyrgavrysh/pneumothorax-binary-classification-task
```

Data Content - > Medical images of lungs done by radiologist during chest x-ray of the patients


### Arguments
Usage:
```
train.py [-h] [-m {B0,B4}] -v VERSION [-e EPOCHS] save_model make_plots

specify arguments of training

positional arguments:
  save_model            saves trained model
  make_plots            plots of losses and accuracy are stored

optional arguments:
  -h, --help            show this help message and exit
  -m {B0,B4}, --model {B0,B4}
                        select models of efficient-net
  -v VERSION, --version VERSION
                        specify version e.g v0 v1 v2...
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs for training
                        
```
### Train Model
```

python3 src/model/train.py -v v0 -m B0 -e 20 True True

```
