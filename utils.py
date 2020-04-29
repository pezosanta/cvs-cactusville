import torch
import numpy as np

cat_list = [
    'Bump', 'Bumpy road', 'Bus stop', 'Children', 'Crossing (blue)', 'Crossing (red)', 'Cyclists', 'Danger (other)', 'Dangerous left turn',
    'Dangerous right turn', 'Give way', 'Go ahead', 'Go ahead or left', 'Go ahead or right', 'Go around either way', 'Go around left', 'Go around right',
    'Intersection', 'Limit 100', 'Limit 120', 'Limit 20', 'Limit 30', 'Limit 50', 'Limit 60', 'Limit 70', 'Limit 80', 'Limit 80 over', 'Limit over',
    'Main road', 'Main road over', 'Multiple dangerous turns', 'Narrow road (left)', 'Narrow road (right)', 'No entry', 'No entry (both directions)',
    'No entry (truck)', 'No stopping', 'No takeover', 'No takeover (truck)', 'No takeover (truck) end', 'No takeover end', 'No waiting', 'One way road',
    'Parking', 'Road works', 'Roundabout', 'Slippery road', 'Stop', 'Traffic light', 'Train crossing', 'Train crossing (no barrier)', 'Wild animals',
    'X - Priority', 'X - Turn left', 'X - Turn right' ]

def add_lists_elementwise(list1, list2):
    array1 = np.array(list1)
    array2 = np.array(list2)

    sum_list = list(array1 + array2)

    return sum_list

def calculate_batch_accuracy(predictions, annotations, batch_size):

    pred_confidence, pred_index = torch.max(predictions, dim = 1)
    gt_confidence, gt_index     = torch.max(annotations, dim = 1)
    
    batch_correct_preds         = torch.eq(pred_index, gt_index).long().sum().item()
    batch_accuracy              = (batch_correct_preds / batch_size) * 100

    # Calculating number of classwise correct predictions
    classwise_correct_preds     = torch.zeros(55).long()

    correct_preds_class         = pred_index[torch.eq(pred_index, gt_index)].long()

    for element in correct_preds_class:
        classwise_correct_preds[element] += 1

    classwise_correct_preds     = classwise_correct_preds.tolist()

    return batch_accuracy, batch_correct_preds, classwise_correct_preds

def calculate_epoch_accuracy(running_correct_preds, running_correct_classwise_preds, num_total_images, num_category_images):
    epoch_accuracy = (running_correct_preds / num_total_images) * 100

    classwise_accuracy = list((np.array(running_correct_classwise_preds) / num_category_images) * 100)

    return epoch_accuracy, classwise_accuracy


