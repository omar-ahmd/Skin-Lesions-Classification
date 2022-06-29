import torch
from skimage.io import imread
from torch.utils import data
import time
from copy import deepcopy
import numpy as np
from PIL import Image
import pandas as pd 

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, max_patience=5, num_epochs=15):
    since = time.time()
    count_iter = 0
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0
    patience=0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            running_corrects_class = np.zeros(7)
            ypred=[]
            y=[]

            # Iterate over data.
            for sample in dataloaders[phase]:
                inputs=sample['image']
                labels=sample['label']
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                if count_iter % 100 == 0:
                    time2 = time.time()
                    print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter, time2 - since, loss/32))
                count_iter += 1

                # statistics
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                ypred += [*list(np.array(preds.cpu()))]
                y += [*list(np.array(labels.data.cpu()))]

                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            dt = pd.DataFrame()
            dt['pred'] = np.array(ypred)
            dt['truth'] = np.array(y)
            dt['weights']= np.array(y)
            for i in range(8):
                dt.loc[dt.index[dt.truth==i], 'weights'] = dt.shape[0]/((dt.truth==i).sum()*8)

            a = (dt.truth == dt.pred)*1
            weighted_acc = (a*dt.weights).sum()/dt.shape[0]
            

            print('{} Loss: {:.4f} Acc: {:.4f} weithed_Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, weighted_acc))

            # deep copy the model
            if phase == 'val' and weighted_acc > best_acc:
                best_acc = weighted_acc
                best_model_wts = deepcopy(model.state_dict())
                patience = 0
            elif phase == 'val':
                patience += 1
                if patience==max_patience:
                    print('Validation is not improving. stopping training')
                    time_elapsed = time.time() - since
                    print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                    print('Best val Acc: {:4f}'.format(best_acc))
                    model.load_state_dict(best_model_wts)
                    return model

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


