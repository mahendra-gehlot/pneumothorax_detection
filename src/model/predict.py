import torch
import os

if os.path.isdir('model/infer_model.pt'):
    model = torch.load('model/infer_model.pt')
    model.eval()
    image_path = 'to_be_added'
    output = model(image_path)
    print(output)

else:
    print('Load model first for inferences!')

