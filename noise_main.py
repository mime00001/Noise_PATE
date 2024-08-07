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

import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import os
from utils import misc, help
import torch.nn as nn

import pandas as pd


LOG_DIR_DATA = "/storage3/michel/data"


def full_run(target_dataset="MNIST", transfer_dataset="FMNIST", nb_teachers=200, params=None, train_teachers=False, compare=True, epsilon=10):
    
    
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
        teachers.util_train_teachers_same_init(dataset_name=target_dataset, n_epochs=75, nb_teachers=nb_teachers, initialize=False) #need to change back to True
    
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


def only_transfer_set(target_dataset="MNIST", transfer_dataset="noise_MNIST", nb_teachers=200, params=None, use_test_loader=False, epsilon=20):
    
    if not params:
        if transfer_dataset=="FMNIST":
            params = {"threshold": 200, "sigma_threshold": 100, "sigma_gnmax": 20, "epsilon": epsilon, "delta" : 1e-5}
        elif target_dataset =="MNIST": 
            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
        elif target_dataset =="CIFAR10": 
            params = {"threshold": 100, "sigma_threshold": 30, "sigma_gnmax": 10, "epsilon": epsilon, "delta" : 1e-5}
        elif target_dataset=="SVHN":
            params = {"threshold": 150, "sigma_threshold": 100, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-6}
        else:
            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}

    noise_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=transfer_dataset, nb_teachers=nb_teachers)
    noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format(transfer_dataset))
    noise_vote_array = noise_vote_array.T
    
    #then perform inference pate
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(transfer_dataset)
    eps, noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
    num_answered = (noise_votes != -1).sum()
    print(len(noise_votes))
    
    #then train the student on Gaussian noise    
    finalacc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset, n_epochs=50, lr=0.001, optimizer="Adam", kwargs=params, use_test_loader=use_test_loader)
    return finalacc, num_answered
    



if __name__ == '__main__':
    #full_run("MNIST", "FMNIST", 200, train_teachers=False)
    
    #teachers.util_train_teachers(dataset_name="CIFAR10", n_epochs=75, nb_teachers=50, initialize=False)
    #full_run("SVHN", "CIFAR10", 250, train_teachers=False)

    #print("A")
    #only_transfer_set("CIFAR10", "noise_CIFAR10", 100, epsilon=5)
    #only_transfer_set("CIFAR10", "noise_CIFAR10", 100, epsilon=10)
    
    #help.run_parameter_search("/vote_array/SVHN.npy", "./pate_CIFAR10forSVHN")
    #for eps in [3, 5, 10]:
    #    help.print_top_values("pate_CIFAR10forSVHN", "num_answered", 5, eps)
    
    #teachers.train_baseline_teacher("CIFAR10", 70)
    #plots.create_kd_data_plot("CIFAR10")
     
    noise_vote_array = pate_data.query_teachers_logits("noise_MNIST", 200)
    noise_vote_array2 = pate_data.query_teachers("noise_MNIST", "MNIST", 200)
    params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 10, "delta" : 1e-5}
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format("noise_MNSIT")
    
    
    acc_log=[]
    acc_sum_softm=[]
    acc_avg_soft=[]
    acc_hist = []
    acc_norm=[]
    
    
    num=[]
    
    for i in range(5):

        pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path)

        a = experiments.use_noisy_softmax_label(40)
        acc_sum_softm.append(a)
        
        a = experiments.use_histogram()
        acc_hist.append(a)
        
        a = experiments.use_ensemble_argmax()
        acc_norm.append(a)
        
        a = experiments.use_logits()
        acc_log.append(a)
        
        a = experiments.use_softmax()
        acc_avg_soft.append(a)


    print(f"summed soft: {np.mean(acc_sum_softm)}")
    print(f"histogram: {np.mean(acc_hist)}")
    print(f"argmax (label): {np.mean(acc_norm)}")
    print(f"logits: {np.mean(acc_log)}")
    print(f"avg softmax: {np.mean(acc_avg_soft)}")
    