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

vote_array_path = LOG_DIR_DATA +  "/vote_array/MNIST.npy"

label_path = LOG_DIR_DATA + "/teacher_labels/MNIST.npy"

def ne():
    
    #vote_array = np.load(vote_array_path)
    #vote_array=vote_array.T
    
    #pate_main.inference_pate(vote_array=vote_array, threshold=150, sigma_threshold=120, sigma_gnmax=40, epsilon=3, delta=10e-8, num_classes=10, savepath=label_path)    
    
    
    #student.util_train_student(dataset_name="MNIST", n_epochs=100, lr=0.001, weight_decay=0, verbose=True, save=True)
    
    #pate_data.create_Gaussian_noise("MNIST", 60000)
    
    #pate_data.query_teachers("noise_MNIST", 200)
    
    noise_vote_array_path = LOG_DIR_DATA +  "/vote_array/noise_MNIST.npy"

    noise_label_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    
    noise_vote_array = np.load(noise_vote_array_path)
    noise_vote_array=noise_vote_array.T
    
    pate_main.inference_pate(vote_array=noise_vote_array, threshold=120, sigma_threshold=110, sigma_gnmax=50, epsilon=5, delta=10e-8, num_classes=10, savepath=noise_label_path) 
    
    noise_votes = np.load(noise_label_path)
    
    other = {"sigma_threshold": 110, "threshold": 120, "sigma_gnmax": 50, "epsilon": 5, "delta" : 10e-8}
    
    unique_v, counts = np.unique(noise_votes, return_counts=True)
    
    for value, count in zip(unique_v, counts):
        print(f"Value {value} occurs {count} times")
    
    student.util_train_student(dataset_name="noise_MNIST", n_epochs=50, lr=0.001, optimizer="Adam", kwargs=other) 
    


#ne()

#for mnist epochs = 35, lr = 0.001, decay = 0, sigma1 = 110, sigma2 = 50, delta = 10e-8, epsilon =3, T=120

LOG_DIR = '/disk2/michel/Pretrained_NW'

teacher_path = LOG_DIR + "/MNIST/teacher_MNIST_mnistresnet.model"

#help.run_parameter_search()
#help.print_top_values("pate_params_new", "num_correctly_answered", 20)

""" accs= []
for i in range(200):
    t = teacher_path + "_{}".format(i)
    accs.append(help.test_model_accuracy(t, "MNIST"))
    print(t  + " accuracy = " + str(accs[i]))

print(np.mean(accs)) """


print(help.test_ensemble_accuracy())
