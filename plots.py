import pate_main, pate_data
import student
from utils import teachers, misc
import conventions
import datasets, models
import time
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torchvision.utils import save_image
from torch.utils.data import DataLoader

LOG_DIR_DATA = "data"
LOG_DIR = ""
LOG_DIR_MODEL = ""

np.set_printoptions(suppress=True)


def final_plot(num_reps=3, target_dataset ="MNIST", 
            query_datasets = ["noise_MNIST", "dead_leaves", "FractalDB", "stylegan", "Shaders21k", "FMNIST", "MNIST"],
            save_path="results/OODness_dictionaries_with_SSL.pkl", nb_teachers=300, student_ssl=True, teacher_ssl=True):
    np.set_printoptions(suppress=True)


    epsilon_range = [6]#[1, 5, 10, 20]
    
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

        
    
def only_transfer_set_different_methods(target_dataset= "MNIST", transfer_dataset="stylegan", nb_teachers=200):
    
    epsilon_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
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
        
            if target_dataset == "MNIST":
                params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
            if target_dataset == "TissueMNIST":
                params = {"threshold": 170, "sigma_threshold": 100, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
                    
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
    
    for num_eps, epsilon in enumerate([1,2,3,4,5,6, 7, 8, 9, 10]):
        if target_dataset == "MNIST":
            params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
        if target_dataset == "TissueMNIST":
            params = {"threshold": 170, "sigma_threshold": 100, "sigma_gnmax": 40, "epsilon": epsilon, "delta" : 1e-5}
                    
        if epsilon == 1:
            noise_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=target_dataset, nb_teachers=nb_teachers, BN_trick=False, SSL=False)
        noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format(target_dataset))
        noise_vote_array = noise_vote_array.T
        
        #then perform inference pate
        noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format(target_dataset)
        eps, noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
        num_answered = (noise_votes != -1).sum()
        print(len(noise_votes))
        
        #then train the student on Gaussian noise    
        finalacc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=target_dataset, n_epochs=50, lr=0.001, optimizer="Adam", kwargs=params)
        results["PATE_MNIST"]["accuracy"][num_eps].append(finalacc)
        results["PATE_MNIST"]["num_answered"][num_eps].append(num_answered)

    
    
    save_path = f"results/differenteps_{target_dataset}_{transfer_dataset}.pkl"
    print(results)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
          
def only_transfer_set_different_methods_CIFAR10(target_dataset= "CIFAR10", transfer_dataset="Shaders21k_CIFAR10"):
    
    epsilon_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    nb_teachers = 50
    
    method = [(False, False, False), (True, False, False), (False, True, True), (True, True, True)]
    
    results= {}
    results["PATE"] = {"accuracy" : [[] for e in epsilon_range], "num_answered" : [[] for e in epsilon_range]}
    results["DataFreeKD"] = {"accuracy" : [[] for e in epsilon_range], "num_answered" : [[] for e in epsilon_range]}
    results["Pretraining"] ={"accuracy" : [[] for e in epsilon_range], "num_answered" : [[] for e in epsilon_range]}
    results["DIET_PATE"] = {"accuracy" : [[] for e in epsilon_range], "num_answered" : [[] for e in epsilon_range]}
    results["PATE_CIFAR10"] = {"accuracy" : [[] for e in epsilon_range], "num_answered" : [[] for e in epsilon_range]}
    
    
    for i, method_specs in enumerate(method):
        bn_trick, ssl_teacher, ssl_student = method_specs
        for num_eps, epsilon in enumerate(epsilon_range):
        

            params = {"threshold": 50, "sigma_threshold": 30, "sigma_gnmax": 15, "epsilon": epsilon, "delta" : 1e-5}

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
                finalacc = student.util_train_SSL_student(target_dataset=target_dataset, transfer_dataset=transfer_dataset,backbone_name="shaders21k_rgb", n_epochs=50, lr=0.001, optimizer="Adam", kwargs=params)
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
    
    for num_eps, epsilon in enumerate([1,2,3,4,5,6,7,8,9,10]):
            params = {"threshold": 50, "sigma_threshold": 30, "sigma_gnmax": 15, "epsilon": epsilon, "delta" : 1e-5}

            if epsilon == 1:
                noise_vote_array = pate_data.query_teachers(target_dataset="CIFAR10", query_dataset="CIFAR10", nb_teachers=nb_teachers, BN_trick=False, SSL=False)
            noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format("CIFAR10"))
            noise_vote_array = noise_vote_array.T
            
            #then perform inference pate
            noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format("CIFAR10")
            eps, noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
            num_answered = (noise_votes != -1).sum()
            print(len(noise_votes))
            
            #then train the student on Gaussian noise    
            finalacc = student.util_train_student(target_dataset="CIFAR10", transfer_dataset="CIFAR10", n_epochs=50, lr=0.001, optimizer="Adam", kwargs=params)
            results["PATE_CIFAR10"]["accuracy"][num_eps].append(finalacc)
            results["PATE_CIFAR10"]["num_answered"][num_eps].append(num_answered)
    
    
    
    save_path = "results/differenteps_cifar10.pkl"
    print(results)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

def plot_count_histogram(title="consensus_same_init.png", votearray_path="/vote_array/noise_MNIST.npy", ylim=0.05, histogram_values_path=None):
    """Experiment, to plot the histograms of the ensemble consensus. The idea is to then train teachers with the same initialization and check if the consensus changes.

    Args:
        title (str, optional): Title of the histogram.
        votearray_path (str, optional): Path to the vote array, which is output by the teachers.
    """
    if votearray_path:
        noise_vote_array = np.load(votearray_path)
        vote_array=noise_vote_array.T
            
        histogram_values = []
        
        for i in range(len(vote_array)):
            uniques, counts = np.unique(vote_array[i], return_counts=True)
            count_most_frequent = counts[np.argmax(counts)]
            
            histogram_values.append(count_most_frequent)
    else:
        histogram_values = np.load(histogram_values_path)
    
    datasetname = votearray_path.split("/")[-1].split(".npy")[0]
        
    plt.hist(histogram_values, bins=240, density=True)
    plt.ylim(0, 0.3)
    plt.xlim(20, 200) 
    plt.ylabel("Occurence")
    plt.xlabel("Number of teachers that agree on final label")
    plt.title(f"Consensus of teachers on {datasetname}")
    plt.savefig(os.path.join(title), dpi=200)
    
    arr = np.array(histogram_values)
    save_path = title.replace(".png", "")
    np.save(os.path.join(save_path), arr)
    

def save_cifar10_images(save_path, num_images=50000):
    os.makedirs(save_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10 size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.CIFAR10(root=LOG_DIR_DATA, train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, (img, _) in enumerate(dataloader):
        torchvision.utils.save_image(img, os.path.join(save_path, f"{i:05d}.png"))
        if i >= num_images - 1:
            break
        
def load_images_from_folder(folder, transform):
    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = torchvision.io.read_image(img_path).float() / 255.0  # Normalize to [0,1]
        img = transform(img)
        images.append(img)
    return torch.stack(images)

def process_images_from_directory(folder, batch_size=50, max_images=1000):
    file_list = sorted(os.listdir(folder))[:max_images]  # Limit number of images
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure correct size
    ])
    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i : i + batch_size]
        images = []
        for filename in batch_files:
            img_path = os.path.join(folder, filename)
            img = torchvision.io.read_image(img_path).float() / 255.0  # Normalize to [0,1]
            img = transform(img)
            images.append(img)
        yield torch.stack(images).to("cuda")

# Function to process images from a .npy file in batches
def process_images_from_npy(npy_file, batch_size=50, max_images=1000):
    images = np.load(npy_file)[:max_images]  # Load and limit images
    images = torch.tensor(images, dtype=torch.float32) / 255.0  # Normalize

    # Ensure shape is (N, 3, H, W)
    if images.ndim == 4 and images.shape[-1] == 3:  # (N, H, W, C) → (N, C, H, W)
        images = images.permute(0, 3, 1, 2)
    elif images.ndim == 4 and images.shape[1] == 3:  # Already in (N, C, H, W)
        pass
    else:  # If grayscale, convert to 3-channel RGB
        images = images.unsqueeze(1).repeat(1, 3, 1, 1)

    # Resize for InceptionV3
    images = torch.nn.functional.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)

    for i in range(0, images.shape[0], batch_size):
        yield images[i : i + batch_size].to("cuda")
        
        
# Save CIFAR-10 images if not already saved
def CIFAR10_KID(real_images_path, generated_images_path):
    if not os.path.exists(real_images_path) or len(os.listdir(real_images_path)) == 0:
        save_cifar10_images(real_images_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize KID metric with a smaller subset
    kid = KernelInceptionDistance(subset_size=50, normalize=True).to(device)


    for real_batch in process_images_from_directory(real_images_path, batch_size=50, max_images=1000):
        kid.update(real_batch, real=True)

# Process generated images from directory or .npy file
    if generated_images_path.endswith(".npy"):
        for gen_batch in process_images_from_npy(generated_images_path, batch_size=50, max_images=1000):
            kid.update(gen_batch, real=False)
    else:
        for gen_batch in process_images_from_directory(generated_images_path, batch_size=50, max_images=1000):
            kid.update(gen_batch, real=False)

    # Compute KID
    kid_mean, kid_std = kid.compute()
    print(f"KID Score: {kid_mean.item():.6f} ± {kid_std.item():.6f}")


def KID_TissueMNIST(real_images_path, generated_images_path, real_npy_file=None, save_real_images=False, batch_size=50, max_images=1000):
    """
    Computes the Kernel Inception Distance (KID) for TissueMNIST grayscale images using MedMNIST library.

    Parameters:
    - real_images_path: Path to the directory containing real images (will be created if saving from .npy).
    - generated_images_path: Path to the directory or .npy file with generated images.
    - real_npy_file: (Optional) Path to a .npy file containing real TissueMNIST images.
    - save_real_images: (Optional) If True, extracts real images from .npy and saves them as individual PNGs.
    - batch_size: Size of each batch during processing.
    - max_images: Maximum number of images to use for KID computation.
    """

    # Ensure the real images directory exists
    if save_real_images and real_npy_file:
        os.makedirs(real_images_path, exist_ok=True)
        real_images = np.load(real_npy_file)  # Load real images
        real_images = torch.tensor(real_images, dtype=torch.float32) / 255.0  # Normalize
        
        # Ensure shape is (N, 1, 28, 28)
        if real_images.ndim == 4 and real_images.shape[-1] == 1:  # (N, H, W, 1) → (N, 1, H, W)
            real_images = real_images.permute(0, 3, 1, 2)
        elif real_images.ndim == 3:  # If missing channel dimension
            real_images = real_images.unsqueeze(1)
        
        # Save images as PNGs
        for i, img in enumerate(real_images):
            save_image(img, os.path.join(real_images_path, f"real_{i}.png"))

    # Device setup (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize KID metric
    kid = KernelInceptionDistance(subset_size=batch_size, normalize=True).to(device)

    # TissueMNIST Dataset Setup using MedMNIST with transformation
    def load_tissuemnist_data():
        # Use MedMNIST to load TissueMNIST with a transformation to Tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = TissueMNIST(root=LOG_DIR_DATA, split='train', download=True, transform=transform)
        return dataset

    # Process real images from MedMNIST
    dataset = load_tissuemnist_data()
    real_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, (images, labels) in enumerate(real_loader):
        if i * batch_size >= max_images:
            break

        # Convert grayscale images (1 channel) to 3 channels (RGB)
        images = images.repeat(1, 3, 1, 1)  # Repeat grayscale across 3 channels
        images = images.to(device)  # Move images to device
        kid.update(images, real=True)

    # Process generated images from directory or .npy file
    def process_images_from_npy(npy_file, batch_size=50, max_images=1000):
        images = np.load(npy_file)[:max_images]  # Load and limit images
        images = torch.tensor(images, dtype=torch.float32) / 255.0  # Normalize

        # Ensure shape is (N, 1, 28, 28)
        if images.ndim == 4 and images.shape[-1] == 1:  # (N, H, W, 1) → (N, 1, H, W)
            images = images.permute(0, 3, 1, 2)
        elif images.ndim == 3:  # If missing channel dimension
            images = images.unsqueeze(1)

        for i in range(0, images.shape[0], batch_size):
            yield images[i : i + batch_size].to(device)

    if generated_images_path.endswith(".npy"):
        for gen_batch in process_images_from_npy(generated_images_path, batch_size=batch_size, max_images=max_images):
            # Convert grayscale images (1 channel) to 3 channels (RGB)
            gen_batch = gen_batch.repeat(1, 3, 1, 1)  # Repeat grayscale across 3 channels
            kid.update(gen_batch, real=False)
    else:
        for gen_batch in process_images_from_directory(generated_images_path, batch_size=batch_size, max_images=max_images):
            # Convert grayscale images (1 channel) to 3 channels (RGB)
            gen_batch = gen_batch.repeat(1, 3, 1, 1)  # Repeat grayscale across 3 channels
            kid.update(gen_batch, real=False)

    # Compute KID
    kid_mean, kid_std = kid.compute()
    print(f"KID Score: {kid_mean.item():.6f} ± {kid_std.item():.6f}")
    return kid_mean.item(), kid_std.item()
