import pate_main, pate_data
import experiments
import student
from utils import teachers, misc
import conventions
import datasets, models
import distill_gaussian

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
import torchvision.transforms as transforms


LOG_DIR_DATA = "/storage3/michel/data"
LOG_DIR = "/storage3/michel"
LOG_DIR_MODEL = "/storage3/michel"


def compare_datasets_BN_trick():
    #create a plot where the method is deployed on different datasets. set the privacy cost to be epsilon=10
    #the datasets should be ordered by accuracy when NOT using the BN trick
    #draw a line through the different accuracies
    #                                                       o
    #                           o             o             x
    #       o                   
    #                           x             x
    #       x
    #   Gaussian noise      Dead Leaves     FMNIST      MNIST pub
    #
    # should look somewhat like this
    
    
    #first query teachers without BN_trick 
    #also should reduce the amount of Gaussian noise back to 60.000
    epsilon = 10
    target_dataset = "MNIST"
    query_datasets = ["noise_MNIST", "dead_leaves", "FractalDB", "StyleGAN", "FMNIST", "MNIST"]
    
    accuracies_wo_BN_trick = {}
    accuracies_with_BN_trick = {}
    
    for ds in query_datasets:
        accuracies_wo_BN_trick[ds] = 0
        accuracies_with_BN_trick[ds] = 0
    
    
    pate_data.create_Gaussian_noise(target_dataset, 60000)   
    params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 5, "delta" : 1e-5}
    fmnist_params = {"threshold": 200, "sigma_threshold": 100, "sigma_gnmax": 20, "epsilon": 5, "delta" : 1e-5} 
    
    #then train the student on the data labeled without BN_trick
    
    for ds in query_datasets:
        vote_array = pate_data.query_teachers(target_dataset, ds, 200, False)
        vote_array = vote_array.T
        label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(ds)
        
        if ds == "FMNIST":
            achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=fmnist_params["threshold"], sigma_threshold=fmnist_params["sigma_threshold"], sigma_gnmax=fmnist_params["sigma_gnmax"], epsilon=epsilon, delta=fmnist_params["delta"], num_classes=10, savepath=label_path)
        else:
            achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=epsilon, delta=params["delta"], num_classes=10, savepath=label_path)
        final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=target_dataset, n_epochs=50)
        
        accuracies_wo_BN_trick[ds] = final_acc
    
    
    for ds in query_datasets:
        vote_array = pate_data.query_teachers(target_dataset, ds, 200, True)
        vote_array = vote_array.T
        label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(ds)
        
        if ds == "FMNIST":
            achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=fmnist_params["threshold"], sigma_threshold=fmnist_params["sigma_threshold"], sigma_gnmax=fmnist_params["sigma_gnmax"], epsilon=epsilon, delta=fmnist_params["delta"], num_classes=10, savepath=label_path)
        else:
            achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=epsilon, delta=params["delta"], num_classes=10, savepath=label_path)
        final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=target_dataset, n_epochs=50)
        
        accuracies_with_BN_trick[ds] = final_acc
    
    
    #sort the accuracies without batchnorm trick in ascending order, then sort the accuracies with batchnorm trick in the same way
    dict(sorted(accuracies_wo_BN_trick.items(), key=lambda item: item[1]))
    accuracies_with_BN_trick= {key: accuracies_with_BN_trick[key] for key in accuracies_wo_BN_trick}
    
    #now just display them
    data = list(accuracies_wo_BN_trick.keys())
    acc_wo = list(accuracies_wo_BN_trick.values())
    acc_with = list(accuracies_with_BN_trick.values())
    
    plt.figure()
    plt.ylim(0, 1)
    
    plt.scatter(data, acc_wo, color="blue", label="Accuracies without BN trick", marker="x")
    plt.plot(data, acc_wo, color="blue", linestyle="dashed")    
    
    plt.scatter(data, acc_with, color="orange", label="Accuracies with BN trick")
    plt.plot(data, acc_with, color="orange", linestyle="dashed")
    
    plt.savefig("OODness_influence.png")
    
    return None