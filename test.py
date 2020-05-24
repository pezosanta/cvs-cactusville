import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import TSDataset
from vgg import vgg16_bn, loadModelParams
from utils import cat_list as cat_list_lowercase

def test():
    test_dataset                        = TSDataset(mode = 'Test')
    test_loader                         = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    checkpoint_path                     = '../../logs/cvs-cactusville/ModelParams/vgg16/vgg16_pretrained-epoch2.pth'
    checkpoint                          = torch.load(checkpoint_path)

    model_state_dict                    = checkpoint['model_state_dict']

    model                               = vgg16_bn(pretrained = False, num_classes = 55)

    model.load_state_dict(state_dict = model_state_dict, strict = True)
    model.to(device = 'cuda:0')
    model.eval()

    current_test_iter = 0
    count = 0

    with torch.no_grad():
        for test_data in test_loader:
            current_test_iter += 1
            if current_test_iter % 100 == 0:
                print('ITER: {} / COUNT: {}'.format(current_test_iter, count))

            images, annotations = test_data

            outs = model(images)
            
            if (torch.argmax(annotations).item() == torch.argmax(outs).item()): count += 1
    
    accuracy = (count / len(test_loader)) * 100
    print('OVERALL ACCURACY: {}'.format(accuracy))

if __name__ == "__main__":
    test()


