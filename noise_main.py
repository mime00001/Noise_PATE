import conventions
import models.resnet10
import models.resnet12
import models.resnet9
import models.mnistresnet
from utils import teachers
from pate_data import query_teachers
import student
import models
import pate_data
import pate_main
import distill_gaussian
#import experiments

import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import os
from utils import misc, help
import torch.nn as nn

import pandas as pd


LOG_DIR_DATA = "/disk2/michel/data"
    
def full_run(target_dataset="MNIST", transfer_dataset="FMNIST", nb_teachers=200, use_test_loader=True, params=None, train_teachers=False):
    
    
    #params = {"threshold": 150, "sigma_threshold": 50, "sigma_gnmax": 20, "epsilon": 26, "delta" : 1e-5}
    
    if not params:
        params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 26, "delta" : 1e-5}
    
    #first train teachers on dataset
    if train_teachers:
        teachers.util_train_teachers_same_init(dataset_name=target_dataset, n_epochs=75, nb_teachers=nb_teachers)
    
    #then query teachers for student labels
    
    vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=target_dataset, nb_teachers=nb_teachers)
    
    #then perform inference PATE
    
    vote_array=vote_array.T
    label_path = LOG_DIR_DATA + "/teacher_labels/MNIST.npy"
    pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=label_path)
    
    #then train student on the noisy teacher labels for baseline
    
    student.util_train_student(target_dataset=target_dataset, transfer_dataset=target_dataset, n_epochs=100)
    
    #then create Gaussian data
    
    pate_data.create_Gaussian_noise(target_dataset, 60000)
    
    #then get the noisy labels for the noise_MNIST
    
    noise_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=transfer_dataset, nb_teachers=nb_teachers)
    noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format(transfer_dataset))
    noise_vote_array = noise_vote_array.T
    
    #then perform inference pate
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(transfer_dataset)
    noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
    
    
    #then train the student on Gaussian noise    
    student.util_train_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset, n_epochs=60, lr=0.001, optimizer="Adam", kwargs=params, use_test_loader=use_test_loader)


def only_transfer_set(target_dataset="MNIST", transfer_dataset="noise_MNIST", nb_teachers=200, params=None, use_test_loader=True):
    
    if not params:
        params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 26, "delta" : 1e-5}


    #noise_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=transfer_dataset, nb_teachers=nb_teachers)
    noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format(transfer_dataset))
    noise_vote_array = noise_vote_array.T
    
    #then perform inference pate
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(transfer_dataset)
    #noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
    
    
    #then train the student on Gaussian noise    
    finalacc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset, n_epochs=2, lr=0.001, optimizer="Adam", kwargs=params, use_test_loader=use_test_loader)
    return finalacc
    
def create_first_table():
    target_dataset = "MNIST"
    nb_teachers=200
    
    params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 5, "delta" : 1e-5}
    
    #vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=target_dataset, nb_teachers=nb_teachers)
    
    #noise_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset="noise_MNIST", nb_teachers=nb_teachers)
    
    #f_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset="FMNIST", nb_teachers=nb_teachers)
    
    #then perform inference PATE
    vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format("MNIST"))
    
    noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format("noise_MNIST"))
    
    f_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format("FMNIST"))
    
    
    vote_array=vote_array.T
    noise_vote_array = noise_vote_array.T
    f_vote_array = f_vote_array.T
    
    
    label_path = LOG_DIR_DATA + "/teacher_labels/MNIST.npy"
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format("noise_MNIST")
    fmnist_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format("FMNIST")
    epsilon_list = [5, 8, 10, 20]
    
    public_list=[]
    gaussian_list=[]
    FMNIST_list=[]
    
    for eps in epsilon_list:
        #public data
        achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=label_path)
        final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=target_dataset, n_epochs=50)
        public_list.append((achieved_eps, final_acc))
        
        #gaussian noise
        achieved_eps, pate_labels = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=noise_label_path)
        final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset="noise_MNIST", n_epochs=50)
        gaussian_list.append((achieved_eps, final_acc))
        
        #fmnist
        achieved_eps, pate_labels = pate_main.inference_pate(vote_array=f_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=fmnist_label_path)
        final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset="FMNIST", n_epochs=50)
        FMNIST_list.append((achieved_eps, final_acc))
        
        
    print("public data list")
    print(public_list)
    
    print("gaussian list")
    print(gaussian_list)
    
    print("public data list")
    print(FMNIST_list)

def betterFMNIST():
    batch_size = 256
    num_workers = 4
    validation_size = 0.2
    transform_train = transform=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.2860,), (0.3530,)) # normalize inputs
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.2860,), (0.3530,)) # normalize inputs
    ])

    trainset = torchvision.datasets.FashionMNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.FashionMNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    end = int(len(testset)*(1-validation_size))
    
    partition_train = [testset[i] for i in range(end)]
    partition_test = [testset[i] for i in range(end, len(testset))]
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("MNIST")
    labels = [[] for i in range(200)]
    
    testdata = next(iter(train_loader))[0].numpy()
    for i in range(200):
        print("querying teacher {}".format(i))
        teacher_name = conventions.resolve_teacher_name(experiment_config)
        teacher_name += "_{}".format(i)
        LOG_DIR = '/disk2/michel/Pretrained_NW'
        teacher_path = os.path.join(LOG_DIR, "MNIST", teacher_name)
        teacher_nw = torch.load(teacher_path)
        teacher_nw = teacher_nw.to(device)
        
        teacher_nw.train() #set model to training mode, batchnorm trick
        
        testindex = 0
        for data, _ in train_loader:
            if testindex == 0:
                assert np.array_equal(testdata, data.numpy()), "first element is not the same in data, problem with dataloader"
            testindex+=1
            data = data.to(device)
            with torch.no_grad():
                teacher_output = teacher_nw(data)   
            label = np.argmax(teacher_output.cpu().numpy(), axis=1)
            for j in label:
                labels[i].append(j)
    path = LOG_DIR_DATA + "/vote_array/{}".format("FMNIST")
    f_vote_array = np.array(labels)
    f_vote_array = f_vote_array.T

    FMNIST_list=[]
    epsilon_list = [5, 8, 10, 20]
    for eps in epsilon_list:
        achieved_eps, pate_labels = pate_main.inference_pate(vote_array=f_vote_array, threshold=150, sigma_threshold=120, sigma_gnmax=40, epsilon=eps, delta=1e-5, num_classes=10, save=False)
        final_acc = student.util_train_student(target_dataset="MNIST", transfer_dataset="FMNIST", n_epochs=50)
        FMNIST_list.append((achieved_eps, final_acc))

if __name__ == '__main__':
    betterFMNIST()