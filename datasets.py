"""
Scripts to get dataloaders for
MNIST, CustomMNIST(0-1, 5-6), CIFAR10, etc.
"""
from utils import util

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

LOG_DIR_DATA = "/disk2/michel/data"

def getDataloaders(trainset, testset, valid_size, batch_size, num_workers):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader, valid_loader, test_loader


################################################################
# Datasets
################################################################

def get_CIFAR10(batch_size, teacher_id, nb_teachers, valid_size=0.2):
    num_workers = 4
    
    assert int(teacher_id) < int(nb_teachers)

    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49139969, 0.48215842, 0.44653093), (0.24703223,0.24348513, 0.26158784)), #(0.2023, 0.1994, 0.2010)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49421429, 0.4851314, 0.45040911), (0.24665252, 0.24289226, 0.26159238)),
    ])
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.CIFAR10(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test) #, transform=transform_test
    batch_len = int(len(trainset) / nb_teachers)
    assert batch_len >= batch_size
    
    start = teacher_id * batch_len
    end = (teacher_id+1) * batch_len
        
    partition_train = [trainset[i] for i in range(start, end)]

    return getDataloaders(partition_train, testset, 0.0, batch_size, num_workers)  # train_loader, valid_loader, test_loader

def get_CIFAR10_PATE(batch_size, validation_size=0.2):
    
    num_workers = 4

    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49139969, 0.48215842, 0.44653093), (0.24703223,0.24348513, 0.26158784)), #(0.2023, 0.1994, 0.2010)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49421429, 0.4851314, 0.45040911), (0.24665252, 0.24289226, 0.26159238)),
    ])
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.CIFAR10(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test) #, transform=transform_test
    
    end = int(len(testset)*(1-validation_size))
    
    partition_train = [testset[i] for i in range(end)]
    partition_test = [testset[i] for i in range(end, len(testset))]
    
    train_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader

def get_MNIST(batch_size, teacher_id, nb_teachers, valid_size=0.2):
    num_workers = 4
    
    assert int(teacher_id) < int(nb_teachers)

    transform_train = transform=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

    trainset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test) #, transform=transform_test
    batch_len = int(len(trainset) / nb_teachers)
    assert batch_len >= batch_size
    
    start = teacher_id * batch_len
    end = (teacher_id+1) * batch_len
        
    partition_train = [trainset[i] for i in range(start, end)]

    return getDataloaders(partition_train, testset, 0.0, batch_size, num_workers)  # train_loader, valid_loader, test_loader

def get_MNIST_PATE(batch_size, validation_size=0.2):
    num_workers = 4
    transform_train = transform=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

    trainset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    end = int(len(testset)*(1-validation_size))
    
    partition_train = [testset[i] for i in range(end)]
    partition_test = [testset[i] for i in range(end, len(testset))]
    
    train_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, valid_loader, test_loader
    
#TODO get_noise_CIFAR10(...)
#TODO get_MNIST(...)
#TODO get_noise_MNIST(...)
