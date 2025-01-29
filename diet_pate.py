import matplotlib.pyplot as plt
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
import datasets


import pickle
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import os
from utils import misc, help
import torch.nn as nn

import pandas as pd


LOG_DIR_DATA = "/data"
LOG_DIR = ""
LOG_DIR_MODEL = ""


def full_run(target_dataset="MNIST", transfer_dataset="noise_MNIST", backbone_name="stylegan", nb_teachers=200, params=None, SSL_teachers=True, train_teachers=True, epsilon=10, BN_trick=True):
    
    
    if target_dataset == "TissueMNIST":
        num_classes = 8
    else:
        num_classes = 10

    if not params:
        if target_dataset =="MNIST": 
            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
        elif target_dataset =="CIFAR10": 
            params = {"threshold": 80, "sigma_threshold": 50, "sigma_gnmax": 20, "epsilon": epsilon, "delta" : 1e-5}
        elif target_dataset == "TissueMNIST":
            params = {"threshold": 170, "sigma_threshold": 100, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
    
    #first train teachers on dataset
    if train_teachers:
        if SSL_teachers:
            teachers.util_train_teachers_SSL_pretrained(dataset_name=target_dataset, backbone_name=backbone_name, n_epochs=50, nb_teachers=nb_teachers)
        else:
            teachers.util_train_teachers_same_init(dataset_name=target_dataset, n_epochs=50, nb_teachers=nb_teachers, initialize=True) #need to change back to True
    

    #then get the noisy labels for the noise_MNIST
    noise_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=transfer_dataset, nb_teachers=nb_teachers, BN_trick=BN_trick) 
    noise_vote_array = noise_vote_array.T
    
    #then perform inference pate
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(transfer_dataset)
    noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=num_classes, savepath=noise_label_path) 
    
    
    #then train the student on Gaussian noise    
    transfer_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset, n_epochs=50, lr=0.001, optimizer="Adam", kwargs=params)
    
    print(f"Accuracy with transfer dataset: {transfer_acc}")


def only_transfer_set(target_dataset="MNIST", transfer_dataset="noise_MNIST", nb_teachers=200, params=None, epsilon=20, BN_trick=True, backbone_name=None):
    ''' 
    This function does not train the teachers and only performs the transfer from the teachers to student. The teachers have to be pretrained for this.
    Target dataset is the dataset we want to transfer knowledge about. Transfer dataset is the dataset we use to transfer the knowledge
    
    '''



    if not params:
        if target_dataset =="MNIST": 
            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
        elif target_dataset =="CIFAR10": 
            params = {"threshold": 80, "sigma_threshold": 50, "sigma_gnmax": 20, "epsilon": epsilon, "delta" : 1e-5}
        elif target_dataset == "TissueMNIST":
            params = {"threshold": 170, "sigma_threshold": 100, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
    
    noise_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=transfer_dataset, nb_teachers=nb_teachers, BN_trick=BN_trick, SSL=True)
    noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format(transfer_dataset))
    noise_vote_array = noise_vote_array.T
    
    #then perform inference pate
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(transfer_dataset)
    eps, noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
    num_answered = (noise_votes != -1).sum()
    print(len(noise_votes))
    
    #then train the student on Gaussian noise    
    if backbone_name:
        finalacc = student.util_train_SSL_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset,backbone_name=backbone_name, n_epochs=50, lr=0.001, optimizer="Adam", kwargs=params)
    else:
        finalacc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset, n_epochs=30, lr=0.001, optimizer="Adam", kwargs=params)
    return finalacc, num_answered
    



if __name__ == '__main__':
    #backbone names ["dead_leaves", "stylegan", "shaders21k_grey", "shaders21k_rgb"]

    #transfer datasets for the target dataset MNIST: [MNIST, FMNIST, stylegan, Shaders21k, noise_MNIST, dead_leaves, FractalDB]

    #transfer datasets for the target dataset TissueMNIST: [FMNIST, stylegan, Shaders21k, noise_MNIST, dead_leaves, FractalDB, TissueMNIST]

    #transfer datasets for the target dataset CIFAR10: [noise_CIFAR10, dead_leaves_CIFAR10, Shaders21k_CIFAR10, stylegan_CIFAR10]

    datasets.prepare_datasets_for_DIET_PATE()

    full_run(target_dataset="MNIST", transfer_dataset="noise_MNIST", backbone_name="stylegan", nb_teachers=200, SSL_teachers=True, train_teachers=True, compare=False, epsilon=5)