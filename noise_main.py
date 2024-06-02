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
    
def full_run(dataset_name="MNIST"):
    #first train teachers on dataset
    #teachers.util_train_teachers_same_init(dataset_name="MNIST", n_epochs=75, nb_teachers=200)
    
    #then query teachers for student labels
    vote_array = pate_data.query_teachers("MNIST", 200)
    
    #then perform inference PATE
    vote_array=vote_array.T
    label_path = LOG_DIR_DATA + "/teacher_labels/MNIST.npy"
    pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=150, sigma_threshold=120, sigma_gnmax=40, epsilon=3, delta=10e-8, num_classes=10, savepath=label_path)
    
    #then train student on the noisy teacher labels for baseline
    student.util_train_student(dataset_name="MNIST", n_epochs=100)
    
    #then create Gaussian data
    #pate_data.create_Gaussian_noise("MNIST", 60000)
    
    #then get the noisy labels for the noise_MNIST
    noise_vote_array = pate_data.query_teachers("noise_MNIST", 200)
    noise_vote_array = noise_vote_array.T
    
    #then perform inference pate
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=150, sigma_threshold=120, sigma_gnmax=40, epsilon=3, delta=10e-8, num_classes=10, savepath=noise_label_path) 
    
    #then train the student on Gaussian noise
    other = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 3, "delta" : 10e-8}
    student.util_train_student(dataset_name="noise_MNIST", n_epochs=100, lr=0.001, optimizer="Adam", kwargs=other)
    


if __name__ == '__main__':
    print("break")