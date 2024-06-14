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
import distill_gaussian
import plots
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
    
def full_run(target_dataset="MNIST", transfer_dataset="FMNIST", nb_teachers=200, use_test_loader=True, params=None, train_teachers=False, compare=True):
    
    
    #params = {"threshold": 150, "sigma_threshold": 50, "sigma_gnmax": 20, "epsilon": 26, "delta" : 1e-5}
    
    if not params:
        params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 10, "delta" : 1e-5}
    
    #first train teachers on dataset
    if train_teachers:
        teachers.util_train_teachers_same_init(dataset_name=target_dataset, n_epochs=75, nb_teachers=nb_teachers, initialize=True)
    
    if compare:
        #then query teachers for student labels
    
        vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=target_dataset, nb_teachers=nb_teachers)
        
        #then perform inference PATE
        
        vote_array=vote_array.T
        label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(target_dataset)
        pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=label_path)
        
        #then train student on the noisy teacher labels for baseline
        
        data_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=target_dataset, n_epochs=100, loss="xe")
        
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
    transfer_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset, n_epochs=100, lr=0.001, optimizer="Adam", kwargs=params,  loss="xe")
    
    if compare:
        print(f"Accuracy with data: {data_acc} and accuracy with transfer set: {transfer_acc}")
    else:
        print(f"Accuracy with transfer dataset: {transfer_acc}")


def only_transfer_set(target_dataset="MNIST", transfer_dataset="noise_MNIST", nb_teachers=200, params=None, use_test_loader=True):
    
    if not params:
        params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 10, "delta" : 1e-5}


    noise_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=transfer_dataset, nb_teachers=nb_teachers)
    noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format(transfer_dataset))
    noise_vote_array = noise_vote_array.T
    
    #then perform inference pate
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(transfer_dataset)
    #noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
    
    
    #then train the student on Gaussian noise    
    finalacc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset, n_epochs=2, lr=0.001, optimizer="Adam", kwargs=params, use_test_loader=use_test_loader)
    return finalacc
    



if __name__ == '__main__':
    full_run("SVHN", "noise_SVHN", nb_teachers=200, use_test_loader=False, train_teachers=True, compare=True)
