# Pneumothorax Binary Classification task

### Project Tree
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
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
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── prepare_data.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

------------

### Context
A pneumothorax occurs when air leaks into the space between your lung and chest wall. 
This air pushes on the outside of your lung and makes it collapse.  

### Implementation Details
The Effiecient-Net B0 Base model is implemented for classification task.

### Dataset
```
https://www.kaggle.com/datasets/volodymyrgavrysh/pneumothorax-binary-classification-task
```

Data Content - > Medical images of lungs done by radiologist during chest x-ray of the patients


### Code
Call following function to train-test Model.
```
args = dict()
args['version'] = 'v0'
args['model'] = model
args['criterion'] = criterion
args['optimizer'] = optimizer
args['epochs'] = 20
args['plotting'] = False
args['perform_testing'] = True

execute(**args)

```
