import pate_main, pate_data
import experiments
import student
from utils import teachers, misc
import conventions
import datasets, models
import distill_gaussian
import compute_FID
import time
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

np.set_printoptions(suppress=True)

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


def final_plot(num_reps=3, target_dataset ="MNIST", 
            query_datasets = ["noise_MNIST", "dead_leaves", "FractalDB", "stylegan", "Shaders21k", "FMNIST", "MNIST"],
            save_path="results/OODness_dictionaries_with_SSL.pkl", nb_teachers=300, student_ssl=True, teacher_ssl=True):
    np.set_printoptions(suppress=True)


    epsilon_range = [5]#[1, 5, 10, 20]
    
    accuracies_wo_BN_trick = {}
    accuracies_with_BN_trick = {}
    accuracies_wo_BN_trick_std = {}
    accuracies_with_BN_trick_std = {}
    num_answered_wo = {}
    num_answered_with ={}
    
    for ds in query_datasets:
        accuracies_wo_BN_trick[ds] = [[] for e in epsilon_range]
        accuracies_with_BN_trick[ds] = [[] for e in epsilon_range]
        accuracies_wo_BN_trick_std[ds] = [[] for e in epsilon_range]
        accuracies_with_BN_trick_std[ds] = [[] for e in epsilon_range]
        num_answered_wo[ds] = [[] for e in epsilon_range]
        num_answered_with[ds] = [[] for e in epsilon_range]
    
    
    #pate_data.create_Gaussian_noise(target_dataset, 60000)   
    #then train the student on the data labeled without BN_trick
       
    for ds in query_datasets:
        vote_array = pate_data.query_teachers(target_dataset, ds, nb_teachers, False, SSL=teacher_ssl)
        vote_array = vote_array.T
        
        for i in range(num_reps):
            
            for i, eps in enumerate(epsilon_range): 
                params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": eps, "delta" : 1e-5}
                if target_dataset == "TissueMNIST":
                    params = {"threshold": 170, "sigma_threshold": 100, "sigma_gnmax": 40, "epsilon": eps, "delta" : 1e-5}
                    
                
                label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(ds)
                
                achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=label_path)
                num_answered = (pate_labels != -1).sum()
                if student_ssl:
                    final_acc = student.util_train_SSL_student(target_dataset=target_dataset, transfer_dataset=ds, backbone_name="stylegan" ,n_epochs=50)
                else:
                    final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=ds ,n_epochs=50)
                accuracies_wo_BN_trick[ds][i].append(final_acc)
                num_answered_wo[ds][i].append(num_answered)
    
    
    for ds in query_datasets:
        vote_array = pate_data.query_teachers(target_dataset, ds, nb_teachers, True, SSL=teacher_ssl)
        vote_array = vote_array.T
        
        for i in range(num_reps):
            
            for i, eps in enumerate(epsilon_range): 
                params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": eps, "delta" : 1e-5}
                #if ds=="FMNIST":
                #    params = {"threshold": 200, "sigma_threshold": 100, "sigma_gnmax": 20, "epsilon": eps, "delta" : 1e-5}
                    
                label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(ds)
                
                achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=label_path)
                num_answered = (pate_labels != -1).sum()
                if student_ssl:
                    final_acc = student.util_train_SSL_student(target_dataset=target_dataset, transfer_dataset=ds, backbone_name="stylegan" ,n_epochs=50)
                else:
                    final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=ds ,n_epochs=50) 
                accuracies_with_BN_trick[ds][i].append(final_acc)
                num_answered_with[ds][i].append(num_answered)
                
    print(f"Accuracies with BN trick: {accuracies_with_BN_trick}")
    print(f"Accuracies without BN trick: {accuracies_wo_BN_trick}")
    for ds in query_datasets:
        for i, eps in enumerate(epsilon_range):
            accuracies_with_BN_trick_std[ds][i] = np.std(accuracies_with_BN_trick[ds][i])
            accuracies_with_BN_trick[ds][i] = np.mean(accuracies_with_BN_trick[ds][i])
            accuracies_wo_BN_trick_std[ds][i] = np.std(accuracies_wo_BN_trick[ds][i])
            accuracies_wo_BN_trick[ds][i] = np.mean(accuracies_wo_BN_trick[ds][i])
            num_answered_wo[ds][i] = np.mean(num_answered_wo[ds][i])
            num_answered_with[ds][i] = np.mean(num_answered_with[ds][i])
    
    #display them in the table as well
    for key, value in accuracies_wo_BN_trick.items():
        print(f"RS: {key}: {value}")
        
    for key, value in accuracies_with_BN_trick.items():
        print(f"CS: {key}: {value}")
        
    
    
    
    with open(save_path, "wb") as f:
        pickle.dump({"accuracies_wo": accuracies_wo_BN_trick, "accuracies_wo_std": accuracies_wo_BN_trick_std,
                     "accuracies_with": accuracies_with_BN_trick, "accuracies_with_std": accuracies_with_BN_trick_std,
                     "num_answered_wo": num_answered_wo, "num_answered_with": num_answered_with}, f)
    
    

def final_plot_CIFAR(num_reps=3, target_dataset ="CIFAR10", 
            query_datasets = ["noise_CIFAR10", "dead_leaves_CIFAR10", "stylegan_CIFAR10", "Shaders21k_CIFAR10", "CIFAR10"],
            save_path="results/CIFAR10_no_SSL.pkl", nb_teachers=50, teacher_ssl=False, student_ssl=False):
    np.set_printoptions(suppress=True)

    epsilon_range = [10]
    
    accuracies_wo_BN_trick = {}
    accuracies_with_BN_trick = {}
    accuracies_wo_BN_trick_std = {}
    accuracies_with_BN_trick_std = {}
    num_answered_wo = {}
    num_answered_with ={}
    
    for ds in query_datasets:
        accuracies_wo_BN_trick[ds] = [[] for e in epsilon_range]
        accuracies_with_BN_trick[ds] = [[] for e in epsilon_range]
        accuracies_wo_BN_trick_std[ds] = [[] for e in epsilon_range]
        accuracies_with_BN_trick_std[ds] = [[] for e in epsilon_range]
        num_answered_wo[ds] = [[] for e in epsilon_range]
        num_answered_with[ds] = [[] for e in epsilon_range]
    
    
    #pate_data.create_Gaussian_noise(target_dataset, 60000)   
    #then train the student on the data labeled without BN_trick
       
    for ds in query_datasets:
        vote_array = pate_data.query_teachers(target_dataset, ds, nb_teachers, False, SSL=teacher_ssl)
        vote_array = vote_array.T
        
        for i in range(num_reps):
            
            for i, eps in enumerate(epsilon_range): 
                if target_dataset == "CIFAR10":
                    params = {"threshold": 50, "sigma_threshold": 30, "sigma_gnmax": 15, "epsilon": eps, "delta" : 1e-5}
                else:
                    params = {"threshold": 150, "sigma_threshold": 100, "sigma_gnmax": 40, "epsilon": eps, "delta" : 1e-6}
                label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(ds)
                
                achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=label_path)
                num_answered = (pate_labels != -1).sum()
                if student_ssl:    
                    final_acc = student.util_train_SSL_student(target_dataset=target_dataset, transfer_dataset=ds, backbone_name="shaders21k_rgb" ,n_epochs=50)
                else:
                    final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=ds ,n_epochs=50)
                   
                accuracies_wo_BN_trick[ds][i].append(final_acc)
                num_answered_wo[ds][i].append(num_answered)
    
    
    for ds in query_datasets:
        vote_array = pate_data.query_teachers(target_dataset, ds, nb_teachers, True, SSL=teacher_ssl)
        vote_array = vote_array.T
        
        for i in range(num_reps):
            
            for i, eps in enumerate(epsilon_range): 
                if target_dataset == "CIFAR10":
                    params = {"threshold": 50, "sigma_threshold": 30, "sigma_gnmax": 15, "epsilon": eps, "delta" : 1e-5}
                else:
                    params = {"threshold": 150, "sigma_threshold": 100, "sigma_gnmax": 40, "epsilon": eps, "delta" : 1e-6}
                    
                label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(ds)
                
                achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=label_path)
                num_answered = (pate_labels != -1).sum()
                if student_ssl:    
                    final_acc = student.util_train_SSL_student(target_dataset=target_dataset, transfer_dataset=ds, backbone_name="shaders21k_rgb" ,n_epochs=50)
                else:
                    final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=ds ,n_epochs=50)
                accuracies_with_BN_trick[ds][i].append(final_acc)
                print(f"Num answered: {num_answered_with}")
                if num_answered_with:
                    num_answered_with[ds][i].append(num_answered)
                else:
                    print(f"Num answered was None!")
                    num_answered_with[ds][i].append(0)
    
    
    print(f"Accuracies with BN trick: {accuracies_with_BN_trick}")
    print(f"Accuracies without BN trick: {accuracies_wo_BN_trick}")
    for ds in query_datasets:
        for i, eps in enumerate(epsilon_range):
            accuracies_with_BN_trick_std[ds][i] = np.std(accuracies_with_BN_trick[ds][i])
            accuracies_with_BN_trick[ds][i] = np.mean(accuracies_with_BN_trick[ds][i])
            accuracies_wo_BN_trick_std[ds][i] = np.std(accuracies_wo_BN_trick[ds][i])
            accuracies_wo_BN_trick[ds][i] = np.mean(accuracies_wo_BN_trick[ds][i])
            num_answered_wo[ds][i] = np.mean(num_answered_wo[ds][i])
            num_answered_with[ds][i] = np.mean(num_answered_with[ds][i])
    
    #display them in the table as well
    for key, value in accuracies_wo_BN_trick.items():
        print(f"RS: {key}: {value}")
        
    for key, value in accuracies_with_BN_trick.items():
        print(f"CS: {key}: {value}")
        
    
    
    
    with open(save_path, "wb") as f:
        pickle.dump({"accuracies_wo": accuracies_wo_BN_trick, "accuracies_wo_std": accuracies_wo_BN_trick_std,
                     "accuracies_with": accuracies_with_BN_trick, "accuracies_with_std": accuracies_with_BN_trick_std,
                     "num_answered_wo": num_answered_wo, "num_answered_with": num_answered_with}, f)
    print(f"save at: {save_path}")
    


def final_plot_TissueMNIST(num_reps=3, target_dataset ="TissueMNIST", 
            query_datasets = ["noise_MNIST", "dead_leaves", "FractalDB", "stylegan", "Shaders21k", "FMNIST", "TissueMNIST"],
            save_path="results/TissueMNIST_AUC_all_SSL.pkl", nb_teachers=250, student_ssl=True, teacher_ssl=True):
    np.set_printoptions(suppress=True)


    epsilon_range = [10]#[1, 5, 10, 20]
    
    accuracies_wo_BN_trick = {}
    accuracies_with_BN_trick = {}
    accuracies_wo_BN_trick_std = {}
    accuracies_with_BN_trick_std = {}
    auc_wo_BN_trick = {}
    auc_with_BN_trick = {}
    auc_wo_BN_trick_std = {}
    auc_with_BN_trick_std = {}
    
    num_answered_wo = {}
    num_answered_with ={}
    
    for ds in query_datasets:
        accuracies_wo_BN_trick[ds] = [[] for e in epsilon_range]
        accuracies_with_BN_trick[ds] = [[] for e in epsilon_range]
        accuracies_wo_BN_trick_std[ds] = [[] for e in epsilon_range]
        accuracies_with_BN_trick_std[ds] = [[] for e in epsilon_range]
        num_answered_wo[ds] = [[] for e in epsilon_range]
        num_answered_with[ds] = [[] for e in epsilon_range]
        auc_wo_BN_trick[ds] = [[] for e in epsilon_range] 
        auc_with_BN_trick[ds] = [[] for e in epsilon_range]
    
    
    #pate_data.create_Gaussian_noise(target_dataset, 60000)   
    #then train the student on the data labeled without BN_trick
       
    for ds in query_datasets:
        vote_array = pate_data.query_teachers(target_dataset, ds, nb_teachers, False, SSL=teacher_ssl)
        vote_array = vote_array.T
        
        for i in range(num_reps):
            
            for i, eps in enumerate(epsilon_range): 
                params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": eps, "delta" : 1e-5}
                if target_dataset == "TissueMNIST":
                    params = {"threshold": 170, "sigma_threshold": 100, "sigma_gnmax": 40, "epsilon": eps, "delta" : 1e-5}
                    
                
                label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(ds)
                
                achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=label_path)
                num_answered = (pate_labels != -1).sum()
                if student_ssl:
                    final_acc = student.util_train_SSL_student(target_dataset=target_dataset, transfer_dataset=ds, backbone_name="stylegan" ,n_epochs=50)
                else:
                    final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=ds ,n_epochs=50)
                    
                    
                accuracies_wo_BN_trick[ds][i].append(final_acc)
                num_answered_wo[ds][i].append(num_answered)
                final_auc = student.util_compute_student_AUC(target_dataset=target_dataset)
                auc_wo_BN_trick[ds][i].append(final_auc)
    
    for ds in query_datasets:
        vote_array = pate_data.query_teachers(target_dataset, ds, nb_teachers, True, SSL=teacher_ssl)
        vote_array = vote_array.T
        
        for i in range(num_reps):
            
            for i, eps in enumerate(epsilon_range): 
                params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": eps, "delta" : 1e-5}
                #if ds=="FMNIST":
                #    params = {"threshold": 200, "sigma_threshold": 100, "sigma_gnmax": 20, "epsilon": eps, "delta" : 1e-5}
                    
                label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(ds)
                
                achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=label_path)
                num_answered = (pate_labels != -1).sum()
                if student_ssl:
                    final_acc = student.util_train_SSL_student(target_dataset=target_dataset, transfer_dataset=ds, backbone_name="stylegan" ,n_epochs=50)
                else:
                    final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=ds ,n_epochs=50) 
                accuracies_with_BN_trick[ds][i].append(final_acc)
                num_answered_with[ds][i].append(num_answered)
                
                final_auc = student.util_compute_student_AUC(target_dataset=target_dataset)
                auc_with_BN_trick[ds][i].append(final_auc)
                
    print(f"Accuracies with BN trick: {accuracies_with_BN_trick}")
    print(f"Accuracies without BN trick: {accuracies_wo_BN_trick}")
    
    for ds in query_datasets:
        for i, eps in enumerate(epsilon_range):
            accuracies_with_BN_trick_std[ds][i] = np.std(accuracies_with_BN_trick[ds][i])
            accuracies_with_BN_trick[ds][i] = np.mean(accuracies_with_BN_trick[ds][i])
            accuracies_wo_BN_trick_std[ds][i] = np.std(accuracies_wo_BN_trick[ds][i])
            accuracies_wo_BN_trick[ds][i] = np.mean(accuracies_wo_BN_trick[ds][i])
            num_answered_wo[ds][i] = np.mean(num_answered_wo[ds][i])
            num_answered_with[ds][i] = np.mean(num_answered_with[ds][i])
            auc_wo_BN_trick[ds][i] = np.mean(auc_wo_BN_trick[ds][i])
            auc_with_BN_trick[ds][i] = np.mean(auc_with_BN_trick[ds][i])
            auc_wo_BN_trick_std[ds][i] = np.std(auc_wo_BN_trick[ds][i])
            auc_with_BN_trick_std[ds][i] = np.std(auc_with_BN_trick[ds][i])
            
    
    #display them in the table as well
    for key, value in accuracies_wo_BN_trick.items():
        print(f"RS: {key}: {value}")
        
    for key, value in accuracies_with_BN_trick.items():
        print(f"CS: {key}: {value}")
        
    
    
    
    with open(save_path, "wb") as f:
        pickle.dump({"accuracies_wo": accuracies_wo_BN_trick, "accuracies_wo_std": accuracies_wo_BN_trick_std,
                     "accuracies_with": accuracies_with_BN_trick, "accuracies_with_std": accuracies_with_BN_trick_std,
                     "num_answered_wo": num_answered_wo, "num_answered_with": num_answered_with, "auc_wo" : auc_wo_BN_trick,
                     "auc_wo_std" : auc_wo_BN_trick_std, "auc_with": auc_with_BN_trick, "auc_with_std": auc_with_BN_trick_std}, f)
    
    


    


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
            compare_data = compute_FID.prep_FMNIST_test(length)
        elif name == "MNIST pub":
            compare_data = compute_FID.prep_MNIST_test(length)
        else:
            compare_data = compute_FID.prep_dataset(name, length)
        fid_score = compute_FID.calculate_FID(base_data.to("cuda"), compare_data.to("cuda"))
        fid_scores[name] = fid_score
        
    print(fid_scores)
    with open("fid_scores.pkl", "wb") as f:
        pickle.dump(fid_scores, f)

def compare_FID_scores_SVHN(length=500):
    
    data_names = ["SVHN pub", "noise_SVHN", "dead_leaves-mixed_SVHN", "stylegan_SVHN", "Shaders21k_SVHN"]
    
    
    base_data = compute_FID.prep_SVHN_train(length)
    
    fid_scores = {}
    
    for name in data_names:
        if name == "SVHN pub":
            compare_data = compute_FID.prep_SVHN_test(length)
        else:
            compare_data = compute_FID.prep_RGB_dataset(name, length)
        print(f"Currently comparing: {name}")
        fid_score = compute_FID.calculate_FID(base_data.to("cuda"), compare_data.to("cuda"))
        fid_scores[name] = fid_score
        
    print(fid_scores)
    with open("results/fid_scores_SVHN.pkl", "wb") as f:
        pickle.dump(fid_scores, f)

   
def compare_KID_scores(length=500):
    
    data_names = ["MNIST pub", "noise_MNIST", "FMNIST", "dead_leaves-mixed", "stylegan-oriented", "FractalDB", "Shaders21k"]
    
    
    base_data = compute_FID.prep_MNIST_train(length)
    
    kid_scores = {}
    
    for name in data_names:
        if name == "FMNIST":
            compare_data = compute_FID.prep_FMNIST_test(length)
        elif name == "MNIST pub":
            compare_data = compute_FID.prep_MNIST_test(length)
        else:
            compare_data = compute_FID.prep_dataset(name, length)
        kid_score = compute_FID.calculate_KID(base_data.to("cuda"), compare_data.to("cuda"))
        kid_scores[name] = kid_score
        
    print(kid_scores)
    with open("results/kid_scores.pkl", "wb") as f:
        pickle.dump(kid_scores, f)
        
def compare_KID_scores_SVHN(length=500):
    
    data_names = ["SVHN pub", "noise_SVHN", "dead_leaves-mixed_SVHN", "stylegan_SVHN", "Shaders21k_SVHN"]
    
    
    base_data = compute_FID.prep_SVHN_train(length)
    
    kid_scores = {}
    
    for name in data_names:
        if name == "SVHN pub":
            compare_data = compute_FID.prep_SVHN_test(length)
        else:
            compare_data = compute_FID.prep_RGB_dataset(name, length)
        print(f"Currently comparing: {name}")
        kid_score = compute_FID.calculate_KID(base_data.to("cuda"), compare_data.to("cuda"))
        kid_scores[name] = kid_score
        
    print(kid_scores)
    with open("results/kid_scores_SVHN.pkl", "wb") as f:
        pickle.dump(kid_scores, f)
        
def compare_FID_scores_FMNIST(length=500):
    data_names = ["FMNIST pub", "noise_MNIST", "MNIST", "dead_leaves-mixed", "stylegan-oriented", "FractalDB", "Shaders21k"]
    
    
    base_data = compute_FID.prep_FMNIST_train(length)
    
    fid_scores = {}
    
    for name in data_names:
        if name == "FMNIST pub":
            compare_data = compute_FID.prep_FMNIST_test(length)
        elif name == "MNIST":
            compare_data = compute_FID.prep_MNIST_train(length)
        else:
            compare_data = compute_FID.prep_dataset(name, length)
        fid_score = compute_FID.calculate_FID(base_data.to("cuda"), compare_data.to("cuda"))
        fid_scores[name] = fid_score
        
    print(fid_scores)
    with open("results/fid_scores_FMNIST.pkl", "wb") as f:
        pickle.dump(fid_scores, f)
        
def compare_KID_scores_FMNIST(length=500):
    
    data_names = ["FMNIST pub", "noise_MNIST", "MNIST", "dead_leaves-mixed", "stylegan-oriented", "FractalDB", "Shaders21k"]
    
    base_data = compute_FID.prep_FMNIST_train(length)
    
    kid_scores = {}
    
    for name in data_names:
        if name == "FMNIST pub":
            compare_data = compute_FID.prep_FMNIST_test(length)
        elif name == "MNIST":
            compare_data = compute_FID.prep_MNIST_train(length)
        else:
            compare_data = compute_FID.prep_dataset(name, length)
        kid_score = compute_FID.calculate_KID(base_data.to("cuda"), compare_data.to("cuda"))
        kid_scores[name] = kid_score
        
    print(kid_scores)
    with open("results/kid_scores_FMNIST.pkl", "wb") as f:
        pickle.dump(kid_scores, f)
        
def only_transfer_set_different_methods(target_dataset= "MNIST", transfer_dataset="stylegan"):
    
    epsilon_range = [1, 2, 3, 4, 5, 6]
    nb_teachers = 200
    
    method = [(False, False, False), (True, False, False), (False, True, True), (True, True, True)]
    
    results= {}
    results["PATE"] = {"accuracy" : [[] for e in epsilon_range], "num_answered" : [[] for e in epsilon_range]}
    results["DataFreeKD"] = {"accuracy" : [[] for e in epsilon_range], "num_answered" : [[] for e in epsilon_range]}
    results["Pretraining"] ={"accuracy" : [[] for e in epsilon_range], "num_answered" : [[] for e in epsilon_range]}
    results["DIET_PATE"] = {"accuracy" : [[] for e in epsilon_range], "num_answered" : [[] for e in epsilon_range]}
    results["PATE_MNIST"] = {"accuracy" : [[] for e in epsilon_range], "num_answered" : [[] for e in epsilon_range]}
    
    
    for i, method_specs in enumerate(method):
        bn_trick, ssl_teacher, ssl_student = method_specs
        for num_eps, epsilon in enumerate(epsilon_range):
        

            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}

            if epsilon == 1:
                noise_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=transfer_dataset, nb_teachers=nb_teachers, BN_trick=bn_trick, SSL=ssl_teacher)
            noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format(transfer_dataset))
            noise_vote_array = noise_vote_array.T
            
            #then perform inference pate
            noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(transfer_dataset)
            eps, noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
            num_answered = (noise_votes != -1).sum()
            print(len(noise_votes))
            
            #then train the student on Gaussian noise    
            if ssl_student:
                finalacc = student.util_train_SSL_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset,backbone_name="stylegan", n_epochs=50, lr=0.001, optimizer="Adam", kwargs=params)
            else:
                finalacc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset, n_epochs=50, lr=0.001, optimizer="Adam", kwargs=params)
            
            if i == 0:
                results["PATE"]["accuracy"][num_eps].append(finalacc)
                results["PATE"]["num_answered"][num_eps].append(num_answered)
            if i == 1:
                results["DataFreeKD"]["accuracy"][num_eps].append(finalacc)
                results["DataFreeKD"]["num_answered"][num_eps].append(num_answered)
            if i == 2:
                results["Pretraining"]["accuracy"][num_eps].append(finalacc)
                results["Pretraining"]["num_answered"][num_eps].append(num_answered)
            if i == 3:
                results["DIET_PATE"]["accuracy"][num_eps].append(finalacc)
                results["DIET_PATE"]["num_answered"][num_eps].append(num_answered)
    
    for num_eps, epsilon in enumerate(epsilon_range):
            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}

            if epsilon == 1:
                noise_vote_array = pate_data.query_teachers(target_dataset="MNIST", query_dataset="MNIST", nb_teachers=nb_teachers, BN_trick=False, SSL=False)
            noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format(transfer_dataset))
            noise_vote_array = noise_vote_array.T
            
            #then perform inference pate
            noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(transfer_dataset)
            eps, noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
            num_answered = (noise_votes != -1).sum()
            print(len(noise_votes))
            
            #then train the student on Gaussian noise    
            finalacc = student.util_train_student(target_dataset="MNIST", transfer_dataset="MNIST", n_epochs=50, lr=0.001, optimizer="Adam", kwargs=params)
            results["PATE_MNIST"]["accuracy"][num_eps].append(finalacc)
            results["PATE_MNIST"]["num_answered"][num_eps].append(num_answered)
    
    
    
    save_path = "results/differenteps_stylegan.pkl"
    print(results)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    
 
 
def timing_dataloaders(batch_size):
    batch_size = batch_size
    num_workers = 4
    validation_size = 0.1
    

    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

     #, transform=transform_train
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    end = int(len(testset)*(1-validation_size))
    
    partition_train = [testset[i] for i in range(end)]
    partition_test = [testset[i] for i in range(end, len(testset))]
    
    train_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, valid_loader, test_loader
 
def time_inference():
     
    teacher_name = "teacher_MNIST_mnistresnet.model"
    teacher_name += "_{}".format(1)
    
    teacher_path = os.path.join("/storage3/michel/OldPretrained_NW/SL_MNIST", teacher_name)
    
    teacher_nw = torch.load(teacher_path)
    teacher_nw = teacher_nw.to("cuda")
    
    for batch_size in [1, 32]:
        train_loader, _, _ = timing_dataloaders(batch_size=batch_size)
        
        all_times = []

        for data, target in train_loader:
            start_time = time.time()
            data, target = data.to("cuda"), target.to("cuda")
            output = teacher_nw(data)
            end_time = time.time()
            
            elapsed = start_time - end_time
            all_times.append(elapsed)
            
        print(f"Average time per batch_size {batch_size}: {np.mean(all_times)} +- {np.std(all_times)}")
        