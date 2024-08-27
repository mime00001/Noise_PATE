import os
import fire
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam, SGD
import torch.nn.functional as F

import models
import datasets
import conventions
import models.resnet9
from utils import misc

# this code is taken from https://github.com/Piyush-555/GaussianDistillation/tree/main


LOG_DIR_DATA = "data"
LOG_DIR_MODEL = ""
LOG_DIR = ""

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
        with torch.no_grad():
            output = target_nw(data)
        loss = criterion(output, target)
        valid_loss += loss.item()
        accs.append(misc.accuracy_metric(output.detach(), target))
    valid_acc = np.mean(accs)

    scheduler.step(train_loss/len(train_loader))
    
    return train_loss/len(train_loader), train_acc, valid_loss/len(valid_loader), valid_acc



def train_teacher(teacher_nw, teacher_id, nb_teachers, train_loader, valid_loader, n_epochs, lr, weight_decay, verbose, device, save, LOG_DIR):
    criterion = nn.CrossEntropyLoss()
    #optimizer = SGD(teacher_nw.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optimizer = Adam(teacher_nw.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=10)
    metrics = []
    for epoch in range(1, n_epochs+1):
        args = train_one_epoch(teacher_nw, train_loader, valid_loader, optimizer, criterion, scheduler, device)
        if verbose:
            print("Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}".format(epoch, *args))
        metrics.append(args)
        
    return [list(i) for i in zip(*metrics)]

@misc.log_experiment
def util_train_teachers(dataset_name, n_epochs, nb_teachers=50, lr=1e-3, weight_decay=0, verbose=True, save=True, LOG_DIR='/storage3/michel/', **kwargs):
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset(dataset_name)
    # override
    for k, v in kwargs.items():
        experiment_config[k] = v
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs(LOG_DIR_DATA, exist_ok=True)
    os.makedirs(LOG_DIR_MODEL + "/Pretrained_NW/{}".format(dataset_name), exist_ok=True)
    
    for teacher_id in range(nb_teachers):
        train_loader, _, valid_loader = eval("datasets.get_{}({}, {}, {})".format(
            dataset_name,
            experiment_config['batch_size'],
            teacher_id,
            nb_teachers
        ))
        teacher_model = model = eval("models.{}.Target_Net({}, {})".format(
            experiment_config['model_teacher'],
            experiment_config['inputs'],
            experiment_config['code_dim']
        )).to(device)
        metrics = train_teacher(teacher_model, teacher_id, nb_teachers,  train_loader, valid_loader, n_epochs, lr, weight_decay, verbose, device, save, LOG_DIR)
        
        model_name = conventions.resolve_teacher_name(experiment_config)
        if nb_teachers!=1:
            model_name+="_{}".format(teacher_id)
        torch.save(model, os.path.join(LOG_DIR_MODEL, 'Pretrained_NW/{}'.format(dataset_name), model_name))
        
        plt.plot(range(1, len(metrics[1])+1), metrics[1], label="Train Accuracy")
        plt.plot(range(1, len(metrics[3])+1), metrics[3], label="Valid Accuracy")
        plt.title('Teacher Training teacher{}'.format(teacher_id))
        plt.legend()
        plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy_teacher{}.png'.format(teacher_id)), dpi=200)
        plt.close()

        plt.plot(range(1, len(metrics[0])+1), metrics[0], label="Train Loss")
        plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Valid Loss")

        plt.title('Teacher Training teacher{}'.format(teacher_id))
        plt.legend()
        plt.savefig(os.path.join(LOG_DIR, 'Plots', 'loss_teacher{}.png'.format(teacher_id)), dpi=200)
        plt.close()
        print("Teacher {} training is finished.".format(teacher_id))
        
@misc.log_experiment        
def train_specific_teacher(teacher_id, dataset_name, n_epochs, nb_teachers=50, lr=1e-3, weight_decay=0, verbose=True, save=True, LOG_DIR='/storage3/michel/', **kwargs):
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset(dataset_name)
    # override
    for k, v in kwargs.items():
        experiment_config[k] = v
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs(LOG_DIR_DATA, exist_ok=True)
    os.makedirs(LOG_DIR_MODEL + '/Pretrained_NW/{}'.format(dataset_name), exist_ok=True)
    train_loader, _, valid_loader = eval("datasets.get_{}({}, {}, {})".format(
            dataset_name,
            experiment_config['batch_size'],
            teacher_id,
            nb_teachers
        ))
    teacher_model = model = eval("models.{}.Target_Net({}, {})".format(
        experiment_config['model_teacher'],
        experiment_config['inputs'],
        experiment_config['code_dim']
    )).to(device)
    metrics = train_teacher(teacher_model, teacher_id, nb_teachers,  train_loader, valid_loader, n_epochs, lr, weight_decay, verbose, device, save, LOG_DIR)
    
    model_name = conventions.resolve_teacher_name(experiment_config)
    model_name+="_{}".format(teacher_id)
    torch.save(model, os.path.join(LOG_DIR_MODEL, 'Pretrained_NW/{}'.format(dataset_name), model_name))
    plt.plot(range(1, len(metrics[1])+1), metrics[1], label="Train Accuracy")
    plt.plot(range(1, len(metrics[3])+1), metrics[3], label="Valid Accuracy")
    plt.title('Teacher Training teacher{}'.format(teacher_id))
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy_teacher{}.png'.format(teacher_id)), dpi=200)
    plt.close()

    plt.plot(range(1, len(metrics[0])+1), metrics[0], label="Train Loss")
    plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Valid Loss")

    plt.title('Teacher Training teacher{}'.format(teacher_id))
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'loss_teacher{}.png'.format(teacher_id)), dpi=200)
    plt.close()
    
    
    
@misc.log_experiment
def util_train_teachers_same_init(dataset_name, n_epochs, nb_teachers=50, lr=1e-3, weight_decay=0, verbose=True, save=True, LOG_DIR='/storage3/michel/',initialize =False,**kwargs):
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset(dataset_name)
    # override
    for k, v in kwargs.items():
        experiment_config[k] = v
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs(LOG_DIR_DATA, exist_ok=True)
    os.makedirs(LOG_DIR_MODEL + '/Pretrained_NW/{}'.format(dataset_name), exist_ok=True)
    
    
    try:
        model = torch.load(os.path.join(LOG_DIR_MODEL, 'Pretrained_NW/{}'.format(dataset_name), "init_model")).to(device)
    except:
        print("init model initialized")
        teacher_model = model = eval("models.{}.Target_Net({}, {})".format(
            experiment_config['model_teacher'],
            experiment_config['inputs'],
            experiment_config['code_dim']
        )).to(device)
        torch.save(model, os.path.join(LOG_DIR_MODEL, 'Pretrained_NW/{}'.format(dataset_name), "init_model"))
    
    if initialize:
        print("init model initialized")
        teacher_model = model = eval("models.{}.Target_Net({}, {})".format(
            experiment_config['model_teacher'],
            experiment_config['inputs'],
            experiment_config['code_dim']
        )).to(device)
        torch.save(model, os.path.join(LOG_DIR_MODEL, 'Pretrained_NW/{}'.format(dataset_name), "init_model"))
    
    for teacher_id in range(nb_teachers):
        train_loader, _, valid_loader = eval("datasets.get_{}({}, {}, {})".format(
            dataset_name,
            experiment_config['batch_size'],
            teacher_id,
            nb_teachers
        ))
        
        teacher_model = model = torch.load(os.path.join(LOG_DIR_MODEL, 'Pretrained_NW/{}'.format(dataset_name), "init_model")).to(device)
        
        metrics = train_teacher(teacher_model, teacher_id, nb_teachers,  train_loader, valid_loader, n_epochs, lr, weight_decay, verbose, device, save, LOG_DIR)
        
        model_name = conventions.resolve_teacher_name(experiment_config)
        if nb_teachers!=1:
            model_name+="_{}".format(teacher_id)
        torch.save(model, os.path.join(LOG_DIR_MODEL, 'Pretrained_NW/{}'.format(dataset_name), model_name))
        
        plt.plot(range(1, len(metrics[1])+1), metrics[1], label="Train Accuracy")
        plt.plot(range(1, len(metrics[3])+1), metrics[3], label="Valid Accuracy")
        plt.title('Teacher Training teacher{}'.format(teacher_id))
        plt.legend()
        plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy_teacher{}.png'.format(teacher_id)), dpi=200)
        plt.close()

        plt.plot(range(1, len(metrics[0])+1), metrics[0], label="Train Loss")
        plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Valid Loss")

        plt.title('Teacher Training teacher{}'.format(teacher_id))
        plt.legend()
        plt.savefig(os.path.join(LOG_DIR, 'Plots', 'loss_teacher{}.png'.format(teacher_id)), dpi=200)
        plt.close()
        print("Teacher {} training is finished.".format(teacher_id))
        
def train_baseline_teacher(dataset_name, n_epochs, lr=1e-3, weight_decay=0, verbose=True, save=True, LOG_DIR='/storage3/michel/', **kwargs):
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset(dataset_name)
    # override
    for k, v in kwargs.items():
        experiment_config[k] = v
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs(LOG_DIR_DATA, exist_ok=True)
    os.makedirs(LOG_DIR_MODEL + '/Pretrained_NW/{}'.format(dataset_name), exist_ok=True)
    train_loader, _, valid_loader = eval("datasets.get_{}({}, {}, {})".format(
            dataset_name,
            experiment_config['batch_size'],
            0,
            1
        ))
    teacher_model = model = eval("models.{}.Target_Net({}, {})".format(
        experiment_config['model_teacher'],
        experiment_config['inputs'],
        experiment_config['code_dim']
    )).to(device)
    metrics = train_teacher(teacher_model, 0, 1,  train_loader, valid_loader, n_epochs, lr, weight_decay, verbose, device, save, LOG_DIR)
    
    model_name = conventions.resolve_teacher_name(experiment_config)
    torch.save(model, os.path.join(LOG_DIR_MODEL, 'Pretrained_NW/{}'.format(dataset_name), model_name))
    plt.plot(range(1, len(metrics[1])+1), metrics[1], label="Train Accuracy")
    plt.plot(range(1, len(metrics[3])+1), metrics[3], label="Valid Accuracy")
    plt.title('Teacher Training teacher')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy_teacher.png'), dpi=200)
    plt.close()

    plt.plot(range(1, len(metrics[0])+1), metrics[0], label="Train Loss")
    plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Valid Loss")

    plt.title('Teacher Training teacher')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'loss_teacher.png'), dpi=200)
    plt.close()