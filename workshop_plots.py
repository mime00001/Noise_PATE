import pate_main, pate_data
import experiments
import student
from utils import teachers, misc
import conventions
import datasets, models
import distill_gaussian
import compute_FID

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps

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
    query_datasets = ["noise_MNIST", "dead_leaves", "FractalDB", "stylegan", "Shaders21k", "FMNIST", "MNIST"]
    
    accuracies_wo_BN_trick = {}
    accuracies_with_BN_trick = {}
    
    for ds in query_datasets:
        accuracies_wo_BN_trick[ds] = 0
        accuracies_with_BN_trick[ds] = 0
    
    
    pate_data.create_Gaussian_noise(target_dataset, 60000)   
    params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 5, "delta" : 1e-5}

    #then train the student on the data labeled without BN_trick
    
    for ds in query_datasets:
        vote_array = pate_data.query_teachers(target_dataset, ds, 200, False)
        vote_array = vote_array.T
        label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(ds)
        
        achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=epsilon, delta=params["delta"], num_classes=10, savepath=label_path)
        final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=ds, n_epochs=50)
        
        accuracies_wo_BN_trick[ds] = final_acc
    
    
    for ds in query_datasets:
        vote_array = pate_data.query_teachers(target_dataset, ds, 200, True)
        vote_array = vote_array.T
        label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(ds)
        
        
        
        achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=epsilon, delta=params["delta"], num_classes=10, savepath=label_path)
        final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=ds, n_epochs=50)
        
        accuracies_with_BN_trick[ds] = final_acc
    
    
    #sort the accuracies without batchnorm trick in ascending order, then sort the accuracies with batchnorm trick in the same way
    accuracies_wo_BN_trick = dict(sorted(accuracies_wo_BN_trick.items(), key=lambda item: item[1]))
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
    
    plt.legend()
    
    plt.savefig("OODness_influence.png")
    
    return None


def final_plot():
    
    epsilon_range = [5, 8, 10, 20]
    target_dataset = "MNIST"
    query_datasets = ["noise_MNIST", "dead_leaves", "FractalDB", "stylegan", "Shaders21k", "FMNIST", "MNIST"]
    
    accuracies_wo_BN_trick = {}
    accuracies_with_BN_trick = {}
    
    for ds in query_datasets:
        accuracies_wo_BN_trick[ds] = [0 for e in epsilon_range]
        accuracies_with_BN_trick[ds] = [0 for e in epsilon_range]
    
    
    #pate_data.create_Gaussian_noise(target_dataset, 60000)   
    #then train the student on the data labeled without BN_trick
       
    for ds in query_datasets:
        vote_array = pate_data.query_teachers(target_dataset, ds, 200, False)
        vote_array = vote_array.T
        for i, eps in enumerate(epsilon_range): 
            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": eps, "delta" : 1e-5}
            if ds=="FMNIST":
                params = {"threshold": 200, "sigma_threshold": 100, "sigma_gnmax": 20, "epsilon": eps, "delta" : 1e-5}
                
            
            label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(ds)
            
            achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=label_path)
            final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=ds, n_epochs=50)
            
            accuracies_wo_BN_trick[ds][i] = final_acc
    
    
    for ds in query_datasets:
        vote_array = pate_data.query_teachers(target_dataset, ds, 200, True)
        vote_array = vote_array.T
        for i, eps in enumerate(epsilon_range): 
            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": eps, "delta" : 1e-5}
            if ds=="FMNIST":
                params = {"threshold": 200, "sigma_threshold": 100, "sigma_gnmax": 20, "epsilon": eps, "delta" : 1e-5}
                
            label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(ds)
            
            achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=label_path)
            final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=ds, n_epochs=50)
            
            accuracies_with_BN_trick[ds][i] = final_acc
        
    
    #display them in the table as well
    for key, value in accuracies_wo_BN_trick.items():
        print(f"RS: {key}: {value}")
        
    for key, value in accuracies_with_BN_trick.items():
        print(f"CS: {key}: {value}")
    
    
    with open("OODness_dictionaries.pkl", "wb") as f:
        pickle.dump({"accuracies_wo": accuracies_wo_BN_trick, "accuracies_with": accuracies_with_BN_trick}, f)
    
    
    
    


def plot_datasets(dataset_name, num=8, spacing=5):
    
    
    original_data_path = LOG_DIR_DATA + "/{}/".format(dataset_name)
    
    original_image = []
    gray_image = []
    
    
    i = 0
    for image in os.listdir(original_data_path):
        original_image.append(Image.open((original_data_path + image))) #.resize((28, 28))
        gray_image.append(ImageOps.grayscale(Image.open((original_data_path + image))).resize((28, 28)))
        i+=1
        if i == num//2: break
        
    #need to be normalized before putting into network               
    total_width = sum(img.width for img in original_image) + (len(original_image) - 1) * spacing

    # Calculate the maximum height for each row
    top_row_height = max(img.height for img in original_image)
    bottom_row_height = max(img.height for img in gray_image)
    total_height = top_row_height + bottom_row_height + spacing

    # Create a blank image with the calculated dimensions
    combined_image = Image.new("RGB", (total_width, total_height), "white")

    # Paste the top row images
    x_offset = 0
    for top_img, bottom_img in zip(original_image, gray_image):
        # Paste the top-row image
       combined_image.paste(top_img, (x_offset, 0))
        
        # Calculate the x_offset to center the bottom image under the top image
       bottom_x_offset = x_offset + (top_img.width - bottom_img.width) // 2
        
        # Paste the bottom image centered under the top image
       y_offset = top_row_height + spacing
       combined_image.paste(bottom_img, (bottom_x_offset, y_offset))
        
        # Update the x_offset for the next column
       x_offset += top_img.width + spacing
    
    save_path = os.path.join(LOG_DIR, "Images", "{}_grid.jpeg".format(dataset_name))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined_image.save(save_path)  


def compare_FID_scores(length=500):
    
    data_names = ["MNIST pub", "noise_MNIST", "FMNIST", "dead_leaves-mixed", "stylegan-oriented", "FractalDB", "Shaders21k"]
    
    
    base_data = compute_FID.prep_MNIST_train(length)
    
    fid_scores = {}
    
    for name in data_names:
        if name == "FMNIST":
            compare_data = compute_FID.prep_FMNIST(length)
        elif name == "MNIST pub":
            compare_data = compute_FID.prep_MNIST_test(length)
        else:
            compare_data = compute_FID.prep_dataset(name, length)
        fid_score = compute_FID.calculate_FID(base_data.to("cuda"), compare_data.to("cuda"))
        fid_scores[name] = fid_score
        
    print(fid_scores)
    with open("fid_scores.pkl", "wb") as f:
        pickle.dump(fid_scores, f)