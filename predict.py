import torch
import cv2
import numpy as np
from dataset import TSDataset
from vgg import vgg16_bn


def loadModelParams():
    checkpoint_path                     = './vgg16_pretrained-epoch2.pth'
    checkpoint                          = torch.load(checkpoint_path)

    model_state_dict                    = checkpoint['model_state_dict']

    model                               = vgg16_bn(pretrained = False, num_classes = 55)

    model.load_state_dict(state_dict = model_state_dict, strict = True)
    model.to(device = 'cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    return model

def predictTS(image, model, convertToRGB):
    if convertToRGB == True:
        image                           = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensorImage                         = torch.unsqueeze(torch.from_numpy(image / 255.0).float().to(device = 'cuda' if torch.cuda.is_available() else 'cpu').permute(2, 0, 1), 0)

    with torch.no_grad():
        outs = model(tensorImage)

        prediction = torch.argmax(outs).item()

    return prediction

    