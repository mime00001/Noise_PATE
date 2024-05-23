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

LOG_DIR_DATA = "/disk2/michel/data"

vote_array_path = LOG_DIR_DATA +  "/vote_array/MNIST.npy"

label_path = LOG_DIR_DATA + "/teacher_labels/MNIST.npy"

def ne():
    
    vote_array = np.load(vote_array_path)
    vote_array=vote_array.T
    
    #pate_main.inference_pate(vote_array=vote_array, threshold=150, sigma_threshold=120, sigma_gnmax=40, epsilon=3, delta=10e-8, num_classes=10, savepath=label_path)    
    
    
    #student.util_train_student(dataset_name="MNIST", n_epochs=100, lr=0.001, weight_decay=0, verbose=True, save=True)
    
    pate_data.create_Gaussian_noise("MNIST", 2000)
    
    pate_data.query_teachers("noise_MNIST", 200)
    
    noise_vote_array_path = LOG_DIR_DATA +  "/vote_array/noise_MNIST.npy"

    noise_label_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    
    noise_vote_array = np.load(noise_vote_array_path)
    noise_vote_array=noise_vote_array.T
    
    pate_main.inference_pate(vote_array=noise_vote_array, threshold=150, sigma_threshold=120, sigma_gnmax=40, epsilon=3, delta=10e-8, num_classes=10, savepath=noise_label_path) 
    
    student.util_train_student(dataset_name="noise_MNIST", n_epochs=100, lr=0.001)  


distill_gaussian.experiment_distil_gaussian("MNIST", 10, 10, compare=True)
 
#ne()

#query_teachers("MNIST", 200)


#kd lr=0.001, decay =0
#CAPC epochs=500, nb_teachers = 50, lr=0.01, decay = 1e-5

#for mnist epochs = 35, lr = 0.001, decay = 0, sigma1 = 120, sigma2 = 40, delta = 10e-8, epsilon =3, T=150


def run_experiment(dataset_name, n_epochs, nb_teachers=200, threshold=150, sigma_threshold=120, sigma_gnmax=40, epsilon=3, delta=10e-8, num_classes=10):
    
    teachers.util_train_teachers(dataset_name=dataset_name, n_epochs=n_epochs, nb_teachers=nb_teachers)
    
    vote_array = pate_data.query_teachers(dataset_name=dataset_name, nb_teachers=nb_teachers).T
    
    pate_main.inference_pate(vote_array=vote_array, threshold=threshold, sigma_threshold=sigma_threshold, sigma_gnmax=sigma_gnmax, epsilon=epsilon, delta=delta, num_classes=num_classes)
    
    student.util_train_student(dataset_name="MNIST", n_epochs=100, lr=0.001, weight_decay=0, verbose=True, save=True)