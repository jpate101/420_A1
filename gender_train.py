# coding:utf8
from torchvision import datasets, models
from torch import nn, optim
from torchvision import transforms as T
from torch.utils import data

import os
import copy
import time
import torch
import os.path
import numpy as np

# data procesing
data_dir = 'gender_data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print(device)
# exit()

# normalize data
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# normalize = T.Normalize([0.5, 0.5, 0.5], [0.225, 0.225, 0.225])
data_transforms = {
    # 'train': T.Compose([
    #     T.RandomResizedCrop(size=(256, 128)),  # resize from middle
    #     T.RandomHorizontalFlip(),  #data flip
    #     # T.RandomVerticalFlip(),
    #     T.ToTensor(),  
    #     normalize
    # ]),
    'train': T.Compose([
        T.Resize(size=(256, 128)),  # resize
        T.CenterCrop(size=(256, 128)),
        # T.Resize(128),  
        # T.CenterCrop(128),
        T.ToTensor(),
        normalize
    ]),

    'val': T.Compose([
        T.Resize(size=(256, 128)),  
        T.CenterCrop(size=(256, 128)),
        # T.Resize(128),  
        # T.CenterCrop(128),
        T.ToTensor(),
        normalize
    ]),
}

# load img
# man label is 0, woman label is 1
image_datasets = {x: datasets.ImageFolder(os.path.join(
    data_dir, x), data_transforms[x]) for x in ['train', 'val']}

# get train and val data amount
dataset_sizes = {x: len(image_datasets[x].imgs) for x in ['train', 'val']}
# dataloaders = {x: data.DataLoader(
#     image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataloaders = {x: data.DataLoader(
    image_datasets[x], batch_size=32, shuffle=True, num_workers=0) for x in ['train', 'val']}
# exit()

# model choice
# model_conv = models.resnet101(pretrained=True)
model_conv = models.resnet34(pretrained=True)
# forze param
# for param in model_conv.parameters():
#     param.requires_grad = False

# extract features from fc layer
fc_features = model_conv.fc.in_features

model_conv.fc = nn.Linear(fc_features, 2)
model_conv.to(device)
# set cross entropy losee
criterion = nn.CrossEntropyLoss()
# define optimizer
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.0001, momentum=0.9)
# optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.0001, momentum=0.9)
optimizer_conv = optim.Adam(model_conv.parameters(),
                            lr=0.0001, betas=(0.9, 0.99))
# learning rate decay
exp_lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer_conv, step_size=25, gamma=0.1)
# exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer_conv, mode='min', verbose=True)



def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()
    # save the best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_iteration = 0


    # loss_meter = meter.AverageValueMeter()
    # confusion_matrix = meter.ConfusionMeter(2)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        temp = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # schduler step
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)   
                labels = labels.to(device)  
                # print('input : ', inputs)
                # print('labels : ', labels)

                # set gradiant as 0
                optimizer.zero_grad()


                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)
                    # print('outputs : ', outputs)
                    
                    _, preds = torch.max(outputs, 1)
                    # print('preds : ', preds)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # scheduler.step(loss) 
                        optimizer.step()

                        # loss_meter.add(loss.item())
                        # confusion_matrix.add(outputs.detach(), labels.detach())

                # statistics
                # cacul `running_loss` and `running_corrects`
                running_loss += loss.item() * inputs.size(0)
                # torch.sum(preds == labels.data)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'train' and epoch_acc > best_train_acc:
                temp = epoch_acc
            if phase == 'val' and epoch_acc > 0 and epoch_acc < temp:
                best_train_acc = temp
                best_val_acc = epoch_acc
                best_iteration = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

    # caculator the time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:4f}'.format(best_iteration))
    print('Best train Acc: {:4f}'.format(best_train_acc))
    print('Best val Acc: {:4f}'.format(best_val_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    model_train = train_model(model_conv, criterion,
                              optimizer_conv, exp_lr_scheduler)
    torch.save(
        model_train, 'GenderTest.pkl')
