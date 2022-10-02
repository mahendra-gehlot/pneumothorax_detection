# Pneumothorax Binary Classification task

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
