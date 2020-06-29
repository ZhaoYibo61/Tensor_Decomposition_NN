from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms  # removed the call to torchvision's ''layers'' and replaced with ours
import os
import copy
import pandas
import time
import numpy

start_time = time.time()


def train_model(model, criterion, optimizer, scheduler, dataloaders_in, dataset_sizes_in, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataloaders = dataloaders_in
    dataset_sizes = dataset_sizes_in

    graph_data_training = pandas.DataFrame()
    graph_data_validation = pandas.DataFrame()

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
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

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # if should_print:
            #     print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'train':
                acc = '{:.4f}'.format(epoch_acc)
                acc = float(acc)
                graph_data_training = graph_data_training.append(
                    pandas.DataFrame([[epoch_loss, acc]], columns=['train_loss', 'train_acc']), ignore_index=True)

            if phase == 'val':
                acc = '{:.4f}'.format(epoch_acc)
                acc = float(acc)
                graph_data_validation = graph_data_validation.append(
                    pandas.DataFrame([[epoch_loss, acc]], columns=['vali_loss', 'vali_acc']), ignore_index=True)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # if should_print: print()

    time_elapsed = time.time() - since
    # if should_print:
    #     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #     print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, graph_data_training, graph_data_validation, pandas.DataFrame([[time_elapsed]],
                                                                               columns=['elapsed_time'])


'''##################################################################################################################'''
for run in range(0, 10):
    '''------------------------------------------------------------------------------------------------------------'''
    # Load Data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    '''------------------------------------------------------------------------------------------------------------'''
    # Found and downloaded the data from https://www.kaggle.com/ajayrana/hymenoptera-data#
    data_dir = '../../hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders_setup = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in
                         ['train', 'val']}
    dataset_sizes_setup = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''-----------------------------------TUCKER RESNET--------------------------------------------------------------'''
    import TNNResnetTucker as tucker_model

    tucker_learner_metrics = pandas.DataFrame()
    tk_model = tucker_model.resnet18(pretrained=False)
    num_ftrs = tk_model.fc.in_features
    tk_model.fc = nn.Linear(num_ftrs, 2)
    tk_model = tk_model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(tk_model.parameters(), lr=0.001, momentum=0.9)  # G: maybe this can go?
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    '''------------------------------------------------------------------------------------------------------------'''
    tk_model, graph_data_training, graph_data_validation, elapsed_time = train_model(tk_model, criterion, optimizer_ft,
                                                                                     exp_lr_scheduler,
                                                                                     dataloaders_in=dataloaders_setup,
                                                                                     dataset_sizes_in=dataset_sizes_setup,
                                                                                     num_epochs=25)
    data = pandas.concat([graph_data_training, graph_data_validation, elapsed_time], axis=1, ignore_index=True)
    tucker_learner_metrics = tucker_learner_metrics.append(data, ignore_index=False)

    '''-----------------------------------NORMAL RESNET--------------------------------------------------------------'''
    from torchvision import models as normal_models

    normal_learner_metrics = pandas.DataFrame()
    normal_resnet_model = normal_models.resnet18(pretrained=False)
    num_ftrs = normal_resnet_model.fc.in_features
    normal_resnet_model.fc = nn.Linear(num_ftrs, 2)
    normal_resnet_model = normal_resnet_model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(normal_resnet_model.parameters(), lr=0.001, momentum=0.9)  # G: maybe this can go?
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    '''------------------------------------------------------------------------------------------------------------'''
    normal_resnet_model, graph_data_training, graph_data_validation, elapsed_time = train_model(normal_resnet_model,
                                                                                                criterion, optimizer_ft,
                                                                                                exp_lr_scheduler,
                                                                                                dataloaders_in=dataloaders_setup,
                                                                                                dataset_sizes_in=dataset_sizes_setup,
                                                                                                num_epochs=25)
    data = pandas.concat([graph_data_training, graph_data_validation, elapsed_time], axis=1, ignore_index=True)
    normal_learner_metrics = normal_learner_metrics.append(data, ignore_index=False)

stamp = '{}_'.format(numpy.random.randint(512, 1024))
file_name = stamp + 'tucker' + '.csv'
tucker_learner_metrics.to_csv(path_or_buf=file_name,
                              header=['train-loss', 'train-acc', 'vali-loss', 'vali-acc', 'elapse-time'])
print('wrote csv file')

stamp = '{}_'.format(numpy.random.randint(1025, 2048))
file_name = stamp + 'normal_resnet' + '.csv'
normal_learner_metrics.to_csv(path_or_buf=file_name,
                              header=['train-loss', 'train-acc', 'vali-loss', 'vali-acc', 'elapse-time'])
print('wrote csv file')

end_time = time.time() - start_time
print(end_time)
print('Total Simulation time completed in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))


