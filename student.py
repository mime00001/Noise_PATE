import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torchvision

import torch.nn as nn
from torch.optim import Adam, SGD
import torch.nn.functional as F

import models
import datasets
import conventions
from utils import misc


def train_one_epoch(target_nw, train_loader, valid_loader, optimizer, criterion, scheduler, device):
    target_nw.train()
    train_loss = 0.0
    accs = []
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = target_nw(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        accs.append(misc.accuracy_metric(output.detach(), target))
    train_acc = np.mean(accs)
    
    target_nw.eval()
    valid_loss = 0.0
    accs = []
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        output = target_nw(data)
        loss = criterion(output, target)
        valid_loss += loss.item()
        accs.append(misc.accuracy_metric(output.detach(), target))
    valid_acc = np.mean(accs)

    scheduler.step(train_loss/len(train_loader))
    
    return train_loss/len(train_loader), train_acc, valid_loss/len(valid_loader), valid_acc



def train_student(student_nw, train_loader, valid_loader, n_epochs, lr, weight_decay, verbose, device, save, LOG_DIR):
    criterion = nn.CrossEntropyLoss()
    #optimizer = SGD(teacher_nw.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optimizer = Adam(student_nw.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=10)
    metrics = []
    for epoch in range(1, n_epochs+1):
        args = train_one_epoch(student_nw, train_loader, valid_loader, optimizer, criterion, scheduler, device)
        if verbose:
            print("Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}".format(epoch, *args))
        metrics.append(args)
        
    return [list(i) for i in zip(*metrics)]





def util_train_student(dataset_name, n_epochs, lr=1e-3, weight_decay=0, verbose=True, save=True, LOG_DIR='/disk2/michel/', **kwargs):
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset(dataset_name)
    # override
    for k, v in kwargs.items():
        experiment_config[k] = v
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs('/disk2/michel/data', exist_ok=True)
    os.makedirs('/disk2/michel/Pretrained_NW/{}'.format(dataset_name), exist_ok=True)
    
    train_loader, _, valid_loader = eval("datasets.get_{}_student({})".format(
        dataset_name,
        experiment_config["batch_size"]
    ))
    student_model = model = eval("models.{}.Target_Net({}, {})".format(
        experiment_config["model_student"],
        experiment_config["inputs"],
        experiment_config["code_dim"]
    )).to(device)
    metrics = train_student(student_model, train_loader, valid_loader, n_epochs, lr, weight_decay, verbose, device, save, LOG_DIR)
       
    model_name = conventions.resolve_student_name(experiment_config)
    torch.save(model, os.path.join('/disk2/michel/Pretrained_NW/{}'.format(dataset_name), model_name))
    
    plt.plot(range(1, len(metrics[1])+1), metrics[1], label="Train Accuracy")
    plt.plot(range(1, len(metrics[3])+1), metrics[3], label="Valid Accuracy")
    plt.title('Student Training')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy_student.png'), dpi=200)
    plt.close()

    plt.plot(range(1, len(metrics[0])+1), metrics[0], label="Train Loss")
    plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Valid Loss")

    plt.title('Student Training')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'loss_student.png'), dpi=200)
    plt.close()
    print("Student training is finished.")
    