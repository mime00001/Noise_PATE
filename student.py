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

LOG_DIR_DATA = "/storage3/michel/data"
LOG_DIR_MODEL = "/storage3/michel"

def train_one_epoch(target_nw, train_loader, valid_loader, optimizer, criterion, scheduler, device, label=True, test_loader=None):
    xe = nn.CrossEntropyLoss()
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
        
        if label: accs.append(misc.accuracy_metric(output.detach(), target))
        train_loss += loss.item()
    train_acc = np.mean(accs)
    
    if test_loader:
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = target_nw(data)   
    
    valid_loss = 0.0
    accs = []
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = target_nw(data)
        loss = xe(output, target)
        valid_loss += loss.item()
        accs.append(misc.accuracy_metric(output.detach(), target))
    valid_acc = np.mean(accs)

    scheduler.step(train_loss/len(train_loader))
    
    return train_loss/len(train_loader), train_acc, valid_loss/len(valid_loader), valid_acc



def train_student(student_nw, train_loader, valid_loader, n_epochs, lr, weight_decay, verbose, device, save, LOG_DIR, optim="Adam", loss="xe", label=False, test_loader=None):
    if loss == "xe":
        criterion = nn.CrossEntropyLoss()
        label=True
    elif loss == "misc":
        criterion= misc.DistillationLoss()
        label=False
    elif loss == "softmax":
        criterion = misc.SoftmaxDistillationLoss()
        label=False
    if optim=="SGD":    
        optimizer = SGD(student_nw.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim=="Adam":
        optimizer = Adam(student_nw.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=10)
    metrics = []
    for epoch in range(1, n_epochs+1):
        args = train_one_epoch(student_nw, train_loader, valid_loader, optimizer, criterion, scheduler, device, test_loader=test_loader,  label=label)
        if verbose:
            print("Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}".format(epoch, *args))
        metrics.append(args)
        
    return [list(i) for i in zip(*metrics)]

@misc.log_experiment
def util_train_student(target_dataset, transfer_dataset, n_epochs, lr=1e-3, weight_decay=0, verbose=True, save=True, LOG_DIR='/disk2/michel/', optimizer="Adam", loss="xe", label=False, **kwargs):
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset(target_dataset)
    # override
    for k, v in kwargs.items():
        experiment_config[k] = v
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs(LOG_DIR_DATA, exist_ok=True)
    os.makedirs(LOG_DIR_MODEL + '/Pretrained_NW/{}'.format(target_dataset), exist_ok=True)
    
    transfer_loader, _, _ = eval("datasets.get_{}_student({})".format(
        transfer_dataset,
        experiment_config["batch_size"]
    ))
    
    _, test_loader, target_loader = eval("datasets.get_{}_student({})".format(
        target_dataset,
        experiment_config["batch_size"]
    ))
    
    
    test_loader =None
    
    student_model = model = eval("models.{}.Target_Net({}, {})".format(
        experiment_config["model_student"],
        experiment_config["inputs"],
        experiment_config["code_dim"]
    )).to(device)
    metrics = train_student(student_model, transfer_loader, target_loader, n_epochs, lr, weight_decay, verbose, device, save, LOG_DIR, optim=optimizer, test_loader=test_loader, loss=loss, label=label)
    
    ret = metrics[3][-1]
    
    model_name = conventions.resolve_student_name(experiment_config)
    torch.save(model, os.path.join(LOG_DIR_MODEL, "Pretrained_NW/{}".format(target_dataset), model_name))
    
    plt.ylim(0, 1)
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
    
    return ret
    
@misc.log_experiment
def util_train_student_same_init(target_dataset, transfer_dataset, n_epochs, lr=1e-3, weight_decay=0, verbose=True, save=True, LOG_DIR='/disk2/michel/', optimizer="Adam", **kwargs):
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset(target_dataset)
    # override
    for k, v in kwargs.items():
        experiment_config[k] = v
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs(LOG_DIR_DATA, exist_ok=True)
    os.makedirs(LOG_DIR_MODEL + '/Pretrained_NW/{}'.format(target_dataset), exist_ok=True)
    
    transfer_loader, _, _ = eval("datasets.get_{}_student({})".format(
        transfer_dataset,
        experiment_config["batch_size"]
    ))
    
    _, test_loader, target_loader = eval("datasets.get_{}_student({})".format(
        target_dataset,
        experiment_config["batch_size"]
    ))
    
    
    test_loader =None
    
    student_model = model = torch.load(os.path.join(LOG_DIR_MODEL, 'Pretrained_NW/MNIST', "init_model"))
    metrics = train_student(student_model, transfer_loader, target_loader, n_epochs, lr, weight_decay, verbose, device, save, LOG_DIR, optim=optimizer, test_loader=test_loader)
    
    
    ret = metrics[3][-1]
    
    model_name = conventions.resolve_student_name(experiment_config)
    torch.save(model, os.path.join(LOG_DIR_MODEL, 'Pretrained_NW/{}'.format(target_dataset), model_name))
    
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
    
    return ret