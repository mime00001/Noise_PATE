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
import experiments
import workshop_plots
import datasets
import compute_FID

import pickle
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import os
from utils import misc, help
import torch.nn as nn

import pandas as pd

LOG_DIR_DATA = "/storage3/michel/data"
LOG_DIR = "/storage3/michel"
LOG_DIR_MODEL = "/storage3/michel"


def full_run(target_dataset="MNIST", transfer_dataset="FMNIST", backbone_name=None, nb_teachers=200, params=None, SSL_teachers=True, train_teachers=False, compare=True, epsilon=10):
    
    
    #params = {"threshold": 150, "sigma_threshold": 50, "sigma_gnmax": 20, "epsilon": 26, "delta" : 1e-5}
    
    if not params:
        if target_dataset =="MNIST": 
            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
        elif target_dataset=="SVHN":
            params = {"threshold": 250, "sigma_threshold": 180, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
        else:
            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
    
    #first train teachers on dataset
    if train_teachers:
        if SSL_teachers:
            teachers.util_train_teachers_SSL_pretrained(dataset_name=target_dataset, backbone_name=backbone_name, n_epochs=50, nb_teachers=nb_teachers)
        else:
            teachers.util_train_teachers_same_init(dataset_name=target_dataset, n_epochs=50, nb_teachers=nb_teachers, initialize=True) #need to change back to True
    
    if compare:
        #then query teachers for student labels
    
        vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=target_dataset, nb_teachers=nb_teachers)
        
        #then perform inference PATE
        
        vote_array=vote_array.T
        label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(target_dataset)
        pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=label_path)
        
        #then train student on the noisy teacher labels for baseline
        
        data_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=target_dataset, n_epochs=50, loss="xe")
        
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
    transfer_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset, n_epochs=50, lr=0.001, optimizer="Adam", kwargs=params,  loss="xe")
    
    if compare:
        print(f"Accuracy with data: {data_acc} and accuracy with transfer set: {transfer_acc}")
    else:
        print(f"Accuracy with transfer dataset: {transfer_acc}")


def only_transfer_set(target_dataset="MNIST", transfer_dataset="noise_MNIST", nb_teachers=200, params=None, epsilon=20, BN_trick=True, backbone_name=None):
    
    if not params:
        #if transfer_dataset=="FMNIST":
         #   params = {"threshold": 200, "sigma_threshold": 100, "sigma_gnmax": 20, "epsilon": epsilon, "delta" : 1e-5}
        if target_dataset =="MNIST": 
            params = {"threshold": 200, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
        elif target_dataset =="CIFAR10": 
            params = {"threshold": 80, "sigma_threshold": 50, "sigma_gnmax": 20, "epsilon": epsilon, "delta" : 1e-5}
        elif target_dataset=="SVHN":
            params = {"threshold": 150, "sigma_threshold": 100, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-6}
        else:
            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}

    #noise_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=transfer_dataset, nb_teachers=nb_teachers, BN_trick=BN_trick)
    noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format(transfer_dataset))
    noise_vote_array = noise_vote_array.T
    
    #then perform inference pate
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(transfer_dataset)
    eps, noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
    num_answered = (noise_votes != -1).sum()
    print(len(noise_votes))
    
    #then train the student on Gaussian noise    
    if backbone_name:
        finalacc = student.util_train_SSL_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset,backbone_name=backbone_name, n_epochs=30, lr=0.001, optimizer="Adam", kwargs=params)
    else:
        finalacc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset, n_epochs=30, lr=0.001, optimizer="Adam", kwargs=params)
    return finalacc, num_answered
    



if __name__ == '__main__':
    #backbone names ["dead_leaves", "stylegan", "shaders21k_grey", "shaders21k_rgb"]
    
    #full_run(target_dataset="CIFAR10", transfer_dataset="noise_CIFAR10", nb_teachers=50, backbone_name="shaders21k_rgb", SSL_teachers=True, train_teachers=True, epsilon=10, compare=True)
    
    #full_run(target_dataset="CIFAR10", transfer_dataset="noise_CIFAR10", nb_teachers=50, SSL_teachers=False, train_teachers=True, epsilon=10, compare=True)
    
    #only_transfer_set("CIFAR10", "noise_CIFAR10", 50, epsilon=10)
    
    target_dataset="CIFAR10"
    backbone_name="shaders21k_rgb"
    nb_teachers=100
    #teachers.util_train_teachers_SSL_pretrained(dataset_name=target_dataset, n_epochs=50, backbone_name=backbone_name,  nb_teachers=nb_teachers)

    #pate_data.query_teachers(target_dataset, "CIFAR10", nb_teachers)
    
    #help.run_parameter_search("/vote_array/Shaders21k_CIFAR10.npy", "/home/michel/project/Noise_PATE/params/100_Shaders21k_CIFAR10.csv")

    #teachers.util_train_teachers_same_init(dataset_name=target_dataset, n_epochs=50, nb_teachers=nb_teachers, initialize=True) #need to change back to True

    #help.print_top_values("params/100_Shaders21k_CIFAR10.csv", "num_correctly_answered", 20, 10)
    #workshop_plots.final_plot_CIFAR(num_reps=3)
    workshop_plots.final_plot(num_reps=3, save_path="results/teachers_not_student_SSL", student_ssl=True, teacher_ssl=False, nb_teachers=200)
    #experiments.plot_count_histogram("plots/consensus_SSL_noise_MNIST.png", "/storage3/michel/data/vote_array/noise_MNIST.npy", ylim=0.3)
    
    #only_transfer_set("CIFAR10", "noise_CIFAR10", 100, epsilon=10, backbone_name="shaders21k_rgb")
    
    #150 120 40 or 200 120 60
    # CIFAR 70 50 30 or 80 50 20