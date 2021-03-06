import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import warnings
from dataset import TSDataset
from vgg import vgg16_bn, loadModelParams
from utils import cat_list as cat_list_lowercase
from utils import add_lists_elementwise, calculate_batch_accuracy, calculate_epoch_accuracy


def train(batch_size = 128, epochs = 150, load_pretrained_weights = True):

    num_train_total_images                  = 104500
    num_val_total_images                    = 5500
    num_train_category_images               = 1900
    num_val_category_images                 = 100

    cat_list                                = [category.upper() for category in cat_list_lowercase]

    base_lr_rate                            = 0.000025
    weight_decay                            = 0.000016

    starting_epoch                          = 0
    starting_train_iter                     = 0
    starting_val_iter                       = 0
    current_status                          = 'Train'

    best_epoch_average_train_loss           = 10000.0   # Updated only if best_epoch_average_val_loss is updated as well
    best_epoch_average_val_loss             = 10000.0
    last_epoch_average_train_loss           = 10000.0       
    last_epoch_average_val_loss             = 10000.0

    best_epoch_train_accuracy               = 0.0
    best_epoch_val_accuracy                 = 0.0
    last_epoch_train_accuracy               = 0.0
    last_epoch_val_accuracy                 = 0.0

    best_epoch_train_classwise_accuracy     = []
    best_epoch_val_classwise_accuracy       = []
    last_epoch_train_classwise_accuracy     = []
    last_epoch_val_classwise_accuracy       = []

    writer_text                             = SummaryWriter('../../logs/cvs-cactusville/Tensorboard/vgg16/vgg16_training_text/')
    writer_avg_train_loss                   = SummaryWriter('../../logs/cvs-cactusville/Tensorboard/vgg16/vgg16_training_avg_train_loss_per_epoch/')
    writer_train_accuracy                   = SummaryWriter('../../logs/cvs-cactusville/Tensorboard/vgg16/vgg16_training_train_accuracy_per_epoch/')
    writer_train_classwise_accuracy         = SummaryWriter('../../logs/cvs-cactusville/Tensorboard/vgg16/vgg16_training_train_classwise_accuracy_per_epoch/')
    writer_avg_valid_loss                   = SummaryWriter('../../logs/cvs-cactusville/Tensorboard/vgg16/vgg16_training_avg_valid_loss_per_epoch/')
    writer_val_accuracy                     = SummaryWriter('../../logs/cvs-cactusville/Tensorboard/vgg16/vgg16_training_valid_accuracy_per_epoch/')
    writer_val_classwise_accuracy           = SummaryWriter('../../logs/cvs-cactusville/Tensorboard/vgg16/vgg16_training_valid_classwise_accuracy_per_epoch/')
    writer_hparams                          = SummaryWriter('../../logs/cvs-cactusville/Tensorboard/vgg16/vgg16_training_hparams/')

    train_dataset                           = TSDataset(mode = 'Train')
    train_loader                            = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    val_dataset                             = TSDataset(mode = 'Val')
    val_loader                              = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    
    model_state_dict                        = None
    optimizer_state_dict                    = None

    if load_pretrained_weights == True:
        checkpoint_path                     = '../../logs/cvs-cactusville/ModelParams/vgg16/vgg16_bn-6c64b313.pth'
        model_state_dict                    = loadModelParams(checkpoint_path)
    else:
        checkpoint_path                     = '../../logs/cvs-cactusville/ModelParams/vgg16/vgg16_pretrained-epoch1.pth'
        checkpoint                          = torch.load(checkpoint_path)

        model_state_dict                    = checkpoint['model_state_dict']
        optimizer_state_dict                = checkpoint['optimizer_state_dict']
        starting_epoch                      = checkpoint['epoch'] + 1                   # Wherever we need to start the training from (based on Tensorboard!)
        starting_train_iter                 = 0#checkpoint['train_iter'] + 1
        starting_val_iter                   = 0#checkpoint['val_iter'] + 1
        base_lr_rate                        = checkpoint['lr']
        weight_decay                        = checkpoint['weight_decay']
        best_epoch_average_train_loss       = checkpoint['best_train_loss']
        best_epoch_average_val_loss         = checkpoint['best_val_loss']
        best_epoch_train_accuracy           = checkpoint['best_train_accuracy']
        best_epoch_val_accuracy             = checkpoint['best_val_accuracy']
        best_epoch_train_classwise_accuracy = checkpoint['best_train_classwise_accuracy']
        best_epoch_val_classwise_accuracy   = checkpoint['best_val_classwise_accuracy']
        last_epoch_average_train_loss       = checkpoint['last_train_loss']
        last_epoch_average_val_loss         = checkpoint['last_val_loss']
        last_epoch_train_accuracy           = checkpoint['last_train_accuracy']
        last_epoch_val_accuracy             = checkpoint['last_val_accuracy']
        last_epoch_train_classwise_accuracy = checkpoint['last_train_classwise_accuracy']
        last_epoch_val_classwise_accuracy   = checkpoint['last_val_classwise_accuracy']

    model                                   = vgg16_bn(pretrained = False, num_classes = 55)    

    criterion                               = nn.BCEWithLogitsLoss()
    optimizer                               = optim.Adam(model.parameters(), lr = base_lr_rate, weight_decay = weight_decay, amsgrad = True)

    model.load_state_dict(state_dict = model_state_dict, strict = False)
    #optimizer.load_state_dict(optimizer_state_dict)

    model.to(device = 'cuda:0')

    writer_text.add_text(tag = 'VGG16/StartingLogs',                                                                                                \
                    text_string = (   'LOADED MODEL PARAMETERS: {}  \n'.format(checkpoint_path)                                                           \
                                    + 'NUMBER OF GPUS: {}  \n'.format(torch.cuda.device_count())                                                          \
                                    + 'BATCH SIZE: {}  \n'.format(batch_size)                                                                             \
                                    + 'NUMBER OF EPOCHS: {}  \n'.format(epochs)                                                                           \
                                    + 'STARTING EPOCH: {}  \n'.format(starting_epoch)                                                                     \
                                    + 'TRAINING ITERATIONS / EPOCH: {}  \n'.format(len(train_loader))                                                     \
                                    + 'VALIDATION ITERATIONS / EPOCH: {}  \n'.format(len(val_loader))                                                     \
                                    + 'LEARNING RATE: {}  \n'.format(base_lr_rate)                                                                        \
                                    + 'WEIGHT DECAY (L2 REGULARIZATION): {}  \n'.format(weight_decay)                                                     \
                                    + '[BEST/LAST] AVERAGE TRAINING LOSS: [{:5f} / {:5f}]  \n'.format(
                                            best_epoch_average_train_loss, last_epoch_average_train_loss)                                                 \
                                    + '[BEST/LAST] AVERAGE VALIDATION LOSS: [{:5f} / {:5f}]  \n'.format(
                                            best_epoch_average_val_loss, last_epoch_average_val_loss)                                                     \
                                    + '[BEST/LAST] TRAINING ACCURACY: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_train_accuracy, last_epoch_train_accuracy)                                                     \
                                    + '[BEST/LAST] VALIDATION ACCURACY: [{:5f} / {:5f}]  \n'.format(
                                            best_epoch_val_accuracy, last_epoch_val_accuracy)),                                                           \
                    global_step = starting_epoch, walltime = None) 

    for current_epoch in range(starting_epoch, epochs):
        writer_text.add_text(tag = 'VGG16/RunningLogs', text_string = 'EPOCH: {}/{}'.format((current_epoch + 1), epochs), global_step = (current_epoch + 1), walltime = None)
               
        epoch_since                             = time.time()

        writer_epoch                            = SummaryWriter('../../logs/cvs-cactusville/Tensorboard/vgg16/vgg16_training_avg_loss_per_iteration_epoch_{}/'.format(current_epoch + 1))

        current_train_iter                      = 0
        current_val_iter                        = 0
        
        running_train_loss                      = 0.0
        current_average_train_loss              = 0.0
        running_val_loss                        = 0.0
        current_average_val_loss                = 0.0

        running_train_correct_preds             = 0
        running_train_correct_classwise_preds   = [0] * 55

        running_val_correct_preds               = 0
        running_val_correct_classwise_preds     = [0] * 55

        is_saved                        = False

        for phase in ['train', 'val']:

            # Train loop
            if phase == 'train':
                train_epoch_since = time.time()

                model.train()

                for train_data in train_loader:
                    
                    current_train_iter += 1

                    images, annotations = train_data               
            
                    outs = model(images)        
            
                    #scheduler = poly_lr_scheduler(optimizer = optimizer, init_lr = base_lr_rate, iter = current_iter, lr_decay_iter = 1, 
                    #                          max_iter = max_iter, power = power)                                                          # max_iter = len(train_loader)
            
                    optimizer.zero_grad()
            
                    loss = criterion(outs, annotations)

                    running_train_loss += loss.item()
                    current_average_train_loss = running_train_loss / current_train_iter

                    batch_accuracy, batch_correct_preds, batch_classwise_correct_preds = calculate_batch_accuracy(outs, annotations, outs.shape[0])
                    running_train_correct_preds += batch_correct_preds
                    running_train_correct_classwise_preds = add_lists_elementwise(running_train_correct_classwise_preds, batch_classwise_correct_preds)

                    writer_epoch.add_scalar(tag = 'VGG16/TrainingIterationAverageLoss'.format(current_epoch + 1), scalar_value = current_average_train_loss, global_step = current_train_iter)
                    writer_epoch.add_scalar(tag = 'VGG16/TrainingIterationAccuracy'.format(current_epoch + 1), scalar_value = batch_accuracy, global_step = current_train_iter)
                    
                    loss.backward(retain_graph = False)
            
                    optimizer.step()
                
                last_epoch_average_train_loss = current_average_train_loss

                last_epoch_train_accuracy, last_epoch_train_classwise_accuracy = calculate_epoch_accuracy(
                    running_train_correct_preds, running_train_correct_classwise_preds, num_train_total_images, num_train_category_images)

                writer_avg_train_loss.add_scalar(tag = 'VGG16/OverfittingLoss', scalar_value = last_epoch_average_train_loss, global_step = (current_epoch + 1))
                writer_train_accuracy.add_scalar(tag = 'VGG16/OverfittingAccuracy', scalar_value = last_epoch_train_accuracy, global_step = (current_epoch + 1))
                
                writer_train_classwise_accuracy.add_scalars(main_tag = 'VGG16/TrainingCategoryAccuracy/', 
                    tag_scalar_dict =  dict(zip(cat_list, last_epoch_train_classwise_accuracy )),
                    global_step = (current_epoch + 1))

                train_time_elapsed = time.time() - train_epoch_since
            
            # Validation loop
            elif phase == 'val':
                val_epoch_since = time.time()   
               
                model.eval()
                
                with torch.no_grad():
                    for val_data in val_loader:
                        
                        current_val_iter += 1
                        
                        images, annotations = val_data                  
                        
                        outs = model(images)
                        
                        val_loss = criterion(outs, annotations)
                       
                        running_val_loss += val_loss.item()
                        current_average_val_loss = running_val_loss / current_val_iter

                        batch_accuracy, batch_correct_preds, batch_classwise_correct_preds = calculate_batch_accuracy(outs, annotations, outs.shape[0])
                        running_val_correct_preds += batch_correct_preds
                        running_val_correct_classwise_preds = add_lists_elementwise(running_val_correct_classwise_preds, batch_classwise_correct_preds)

                        writer_epoch.add_scalar(tag = 'VGG16/ValidationIterationAverageLoss'.format((current_epoch + 1)), scalar_value = current_average_val_loss, global_step = current_val_iter)
                        writer_epoch.add_scalar(tag = 'VGG16/ValidationIterationAccuracy'.format(current_epoch + 1), scalar_value = batch_accuracy, global_step = current_val_iter)

                    last_epoch_average_val_loss = current_average_val_loss

                    last_epoch_val_accuracy, last_epoch_val_classwise_accuracy = calculate_epoch_accuracy(
                        running_val_correct_preds, running_val_correct_classwise_preds, num_val_total_images, num_val_category_images)

                    writer_avg_valid_loss.add_scalar(tag = 'VGG16/OverfittingLoss', scalar_value = last_epoch_average_val_loss, global_step = (current_epoch + 1))
                    writer_val_accuracy.add_scalar(tag = 'VGG16/OverfittingAccuracy', scalar_value = last_epoch_val_accuracy, global_step = (current_epoch + 1))

                    writer_val_classwise_accuracy.add_scalars(main_tag = 'VGG16/ValidationCategoryAccuracy/', 
                        tag_scalar_dict =  dict(zip(cat_list, last_epoch_val_classwise_accuracy )),
                        global_step = (current_epoch + 1))

                    val_time_elapsed = time.time() - val_epoch_since

        # Saving model parameters if average_val_loss or mAP is improved
        if(last_epoch_val_accuracy >= best_epoch_val_accuracy):
            is_saved                            = True
            
            best_epoch_val_accuracy             = last_epoch_val_accuracy
            best_epoch_train_accuracy           = last_epoch_train_accuracy

            best_epoch_val_classwise_accuracy   = last_epoch_val_classwise_accuracy
            best_epoch_train_classwise_accuracy = last_epoch_train_classwise_accuracy

            best_epoch_average_val_loss         = last_epoch_average_val_loss
            best_epoch_average_train_loss       = last_epoch_average_train_loss

            PATH = '../../logs/cvs-cactusville/ModelParams/vgg16/vgg16_pretrained-epoch{}.pth'.format(current_epoch + 1)                        
            torch.save({
                    'epoch': current_epoch,
                    'train_iter': current_train_iter,
                    'val_iter': current_val_iter,
                    'lr': base_lr_rate,
                    'weight_decay': weight_decay,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_train_loss': best_epoch_average_train_loss,
                    'best_val_loss': best_epoch_average_val_loss,
                    'last_train_loss': last_epoch_average_train_loss,
                    'last_val_loss': last_epoch_average_val_loss,
                    'best_train_accuracy': best_epoch_train_accuracy,
                    'best_val_accuracy': best_epoch_val_accuracy,
                    'last_train_accuracy': last_epoch_train_accuracy,
                    'last_val_accuracy': last_epoch_val_accuracy,
                    'best_train_classwise_accuracy': best_epoch_train_classwise_accuracy,
                    'best_val_classwise_accuracy': best_epoch_val_classwise_accuracy,
                    'last_train_classwise_accuracy': last_epoch_train_classwise_accuracy,
                    'last_val_classwise_accuracy': last_epoch_val_classwise_accuracy
                    }, PATH)
                
            writer_text.add_text(tag = 'VGG16/SavingLogs',                                                                     \
                        text_string = (   '!!! IMPROVEMENT !!! MODEL PARAMETERS HAVE BEEN SAVED !!!  \n'                       \
                                        + '[BEST/LAST] AVERAGE TRAINING LOSS: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_average_train_loss, last_epoch_average_train_loss)                  \
                                        + '[BEST/LAST] AVERAGE VALIDATION LOSS: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_average_val_loss, last_epoch_average_val_loss)                      \
                                        + '[BEST/LAST] TRAINING ACCURACY: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_train_accuracy, last_epoch_train_accuracy)                          \
                                        + '[BEST/LAST] VALIDATION ACCURACY: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_val_accuracy, last_epoch_val_accuracy)),                            \
                        global_step = (current_epoch + 1), walltime = None)

        else:
             writer_text.add_text(tag = 'VGG16/SavingLogs',                                                                    \
                        text_string = (   '!!! NO IMPROVEMENT !!!  \n'                                                         \
                                        + '[BEST/LAST] AVERAGE TRAINING LOSS: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_average_train_loss, last_epoch_average_train_loss)                  \
                                        + '[BEST/LAST] AVERAGE VALIDATION LOSS: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_average_val_loss, last_epoch_average_val_loss)                      \
                                        + '[BEST/LAST] TRAINING ACCURACY: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_train_accuracy, last_epoch_train_accuracy)                          \
                                        + '[BEST/LAST] VALIDATION ACCURACY: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_val_accuracy, last_epoch_val_accuracy)),                            \
                        global_step = (current_epoch + 1), walltime = None)
        
        hparams_dict_1 = {  'EPOCH': str(current_epoch + 1),
                            'SAVED': str(is_saved),
                            'BATCH SIZE': str(batch_size),
                            'OPTIMIZER': 'ADAM (AMSGRAD)',
                            'LEARNING RATE': str(base_lr_rate),
                            'WEIGHT DECAY': str(weight_decay),
                            'TRAIN LOSS': '{:5f}'.format(last_epoch_average_train_loss),
                            'VAL LOSS': '{:5f}'.format(last_epoch_average_val_loss)}

        hparams_dict_2 = dict(zip(cat_list, last_epoch_val_classwise_accuracy ))
        '''
        # There is not any add_hparams function in SummaryWriter in torch==1.1.0 (added in torch==1.3.0)
        writer_hparams.add_hparams( hparam_dict = {**hparams_dict_1, **hparams_dict_2},
                                    metric_dict = {'VGG16/W_LEARNING_RATE': base_lr_rate})
        '''
        epoch_time_elapsed = time.time() - epoch_since

        # Epoch ending logs
        writer_text.add_text(tag = 'VGG16/RunningLogs',                                                                        \
                        text_string = (   '[BEST/LAST] AVERAGE TRAINING LOSS: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_average_train_loss, last_epoch_average_train_loss)                  \
                                        + '[BEST/LAST] AVERAGE VALIDATION LOSS: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_average_val_loss, last_epoch_average_val_loss)                      \
                                        + '[BEST/LAST] TRAINING ACCURACY: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_train_accuracy, last_epoch_train_accuracy)                          \
                                        + '[BEST/LAST] VALIDATION ACCURACY: [{:5f} / {:5f}]  \n'.format(
                                                best_epoch_val_accuracy, last_epoch_val_accuracy)                              \
                                        + 'EPOCH TIME: {}:{}  \n'.format(
                                                int(epoch_time_elapsed // 60 // 60), int(epoch_time_elapsed // 60 % 60))       \
                                        + 'TRAINING TIME: {}:{}  \n'.format(
                                                int(train_time_elapsed // 60 // 60), int(train_time_elapsed // 60 % 60))       \
                                        + 'VALIDATION TIME: {}:{}  \n'.format(
                                                int(val_time_elapsed // 60 // 60), int(val_time_elapsed // 60 % 60))),         \
                        global_step = (current_epoch + 1), walltime = None)

if __name__ == "__main__":
    train(batch_size = 64, epochs = 150, load_pretrained_weights = False)
