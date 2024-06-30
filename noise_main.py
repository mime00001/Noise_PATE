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
    
def full_run(target_dataset="MNIST", transfer_dataset="FMNIST", nb_teachers=200, use_test_loader=True, params=None, train_teachers=False, compare=True, epsilon=10):
    
    
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
        teachers.util_train_teachers_same_init(dataset_name=target_dataset, n_epochs=75, nb_teachers=nb_teachers, initialize=True)
    
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


def only_transfer_set(target_dataset="MNIST", transfer_dataset="noise_MNIST", nb_teachers=200, params=None, use_test_loader=False, epsilon=10):
    
    if not params:
        if target_dataset =="MNIST": 
            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
        elif target_dataset=="SVHN":
            params = {"threshold": 300, "sigma_threshold": 200, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-6}
        else:
            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}

    #noise_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=transfer_dataset, nb_teachers=nb_teachers)
    noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format(transfer_dataset))
    noise_vote_array = noise_vote_array.T
    
    #then perform inference pate
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(transfer_dataset)
    noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
    
    
    #then train the student on Gaussian noise    
    finalacc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset, n_epochs=80, lr=0.001, optimizer="Adam", kwargs=params, use_test_loader=use_test_loader)
    return finalacc
    



if __name__ == '__main__':
    params = {"threshold": 190, "sigma_threshold": 100, "sigma_gnmax": 50, "epsilon": 10, "delta" : 1e-6}
    
    only_transfer_set("SVHN", "noise_SVHN", 250, params=None, epsilon=10)
    #help.run_parameter_search()
    #help.print_top_values("pate_params_SVHN", "num_answered", 10, 10)
    
    
    
    """ public_list = []
    gaussian_list =[]
    CIFAR10_list =[]
    
    epsilon = [5, 8, 10, 20]
    
    params = {"threshold": 300, "sigma_threshold": 200, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-6}
    
    
    #vote_array = pate_data.query_teachers(target_dataset="SVHN", query_dataset="SVHN", nb_teachers=250)
    vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format("SVHN")).T
    
    
    
    #noise_vote_array = pate_data.query_teachers(target_dataset="SVHN", query_dataset="noise_SVHN", nb_teachers=250)
    noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format("noise_SVHN")).T
    
    
    #noise_vf_vote_arrayote_array = pate_data.query_teachers(target_dataset="SVHN", query_dataset="CIFAR10", nb_teachers=250)
    f_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format("CIFAR10")).T
    
    target_dataset = "SVHN"
    
    label_path = LOG_DIR_DATA + "/teacher_labels/SVHN.npy"
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format("noise_SVHN")
    fmnist_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format("CIFAR10")
    
    for eps in epsilon:
        
        
        #public data
        achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=label_path)
        final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=target_dataset, n_epochs=50)
        public_list.append((round(achieved_eps, 3), round(final_acc, 3)))
        
        #gaussian noise
        achieved_eps, pate_labels = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=noise_label_path)
        final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset="noise_SVHN", n_epochs=50)
        gaussian_list.append((round(achieved_eps, 3), round(final_acc, 3)))
        
        #fmnist
        achieved_eps, pate_labels = pate_main.inference_pate(vote_array=f_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=fmnist_label_path)
        final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset="CIFAR10", n_epochs=50)
        CIFAR10_list.append((round(achieved_eps, 3), round(final_acc, 3)))
    
    
    headers = ['eps=5', 'eps=8', "eps=10", "eps=20"]
    row_labels = [ "public_data", "Gaussian noise", "CIFAR10 data"]
    values = [
        [public_list[0], public_list[1], public_list[2], public_list[3]],
        [gaussian_list[0], gaussian_list[1], gaussian_list[2], gaussian_list[3]],
        [CIFAR10_list[0], CIFAR10_list[1], CIFAR10_list[2], CIFAR10_list[3]]
    ]
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4)

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=values, colLabels=headers, rowLabels=row_labels, loc='center', cellLoc='center')

    # Adjust layout
    plt.subplots_adjust(left=0.2, top=0.8)
 
    # Save the table to a file
    plt.savefig('table 1_SVHN.png') """