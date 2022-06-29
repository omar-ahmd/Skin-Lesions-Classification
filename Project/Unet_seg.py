import torch
from skimage.io import imread
from torch.utils import data
import time
from copy import deepcopy
import numpy as np
from PIL import Image

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    idx: int):
        # Select the sample
        x = Image.fromarray(np.uint8(self.inputs[idx, :, :, :]))
        
        y = Image.fromarray(np.uint8(255*self.targets[idx, :, :]))
        
        sample = {"image": x, "mask": y}

        # Preprocessing
        if self.transform is not None:
            sample['image'] = self.transform[0](sample['image'])
            sample['mask'] = self.transform[1](sample['mask'])
        return sample
    
    
    
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)



def train_unet_model(model, criterion, optimizer, scheduler, dataloaders, device, max_patience=5, num_epochs=15):
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
            k=0
            # Iterate over data.
            for sample in dataloaders[phase]:
                inputs=sample['image']
                masks=sample['mask']
                inputs = inputs.to(device)
                masks = masks.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                if count_iter % 100 == 0 :
                    time2 = time.time()
                    print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter, time2 - since, loss/32))
                count_iter += 1
                

                # statistics
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += dice_coeff(torch.round(outputs), masks)
                
                k+=1
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / k
            epoch_acc = running_corrects.double() / k

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
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
                    

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

