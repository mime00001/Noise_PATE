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
import torch.nn as nn
import numpy as np
import os
from utils import misc

import pandas as pd


LOG_DIR_DATA = "/disk2/michel/data"


def remove_rows(input_file, output_file, column_name, condition):
    # Read the CSV file
    df = pd.read_csv(input_file)
    

    # Apply the condition to filter out rows
    filtered_df = df[~df[column_name].apply(condition)]
    

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_file, index=False)


def print_top_values(input_file, column_name, top_n_values, target_epsilon=3):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    condition = lambda x : x != target_epsilon
    
    condition2 = lambda x : x <= 0.6
    
    df = df[~df["target_epsilon"].apply(condition)]
    df = df[~df["num_correctly_answered"].apply(condition2)]
    
    # Sort the DataFrame by the specified column in descending order
    sorted_df = df.sort_values(by=column_name, ascending=False)
    
    # Get the top N rows
    top_n_df = sorted_df.head(top_n_values)
    
    # Display the top N rows
    print(f"\nTop {top_n_values} rows by {column_name}:")
    print(top_n_df)


def run_parameter_search():
    
    noise_vote_array_path = LOG_DIR_DATA +  "/vote_array/noise_SVHN.npy"
    
    noise_vote_array = np.load(noise_vote_array_path)
    noise_vote_array=noise_vote_array.T
    
    threshold_list = [190, 250, 300]
    sigma_threshold_list = [100, 150, 200]
    sigma_gnmax_list = [30, 40, 50]
    epsilon_list = [3, 5, 10]
    delta_list =[1e-6]
    num_classes=10
    savepath='./pate_params_SVHN'
    
    predicted_labels = pate_data.get_argmax_labels(noise_vote_array)
    
    pate_main.tune_pate(noise_vote_array, threshold_list, sigma_threshold_list, sigma_gnmax_list, epsilon_list, delta_list, num_classes, savepath, predicted_labels)
    
    
def test_model_accuracy(model_path, dataset_name):
    
    device = misc.get_device()
    
    teacher_nw = torch.load(model_path)
    teacher_nw.to(device)
    teacher_nw.train()
    criterion = nn.CrossEntropyLoss()
    accs = []
    
    train_loader, _, valid_loader = eval("datasets.get_{}({}, {}, {})".format(
        dataset_name,
        256,
        0,
        1
    ))
    
    valid_loss = 0.0
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = teacher_nw(data)
        loss = criterion(output, target)
        valid_loss += loss.item()
        accs.append(misc.accuracy_metric(output.detach(), target))
    valid_acc = np.mean(accs)
    
    print(valid_acc)
    return valid_acc

def test_ensemble_accuracy(dataset_name):
    
    if dataset_name == "noise_MNIST":
        noise_vote_array_path = LOG_DIR_DATA +  "/vote_array/noise_MNIST.npy"

        noise_label_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"

        noise_vote_array = np.load(noise_vote_array_path)
        vote_array=noise_vote_array.T
        t,l,r = datasets.get_noise_MNIST_PATE(256)
        
        true_labels=[]
        device = misc.get_device()
        experiment_config = conventions.resolve_dataset("noise_MNIST")
        teacher_name = conventions.resolve_teacher_name(experiment_config)
        teacher_path = os.path.join("/disk2/michel", "Pretrained_NW","MNIST", teacher_name)
        teacher_nw = torch.load(teacher_path)
        teacher_nw.to(device)
        teacher_nw.train()
        
            
        for data, _ in t:
            data = data.to(device)
            with torch.no_grad():
                teacher_output = teacher_nw(data)   
            label = np.argmax(teacher_output.cpu().numpy(), axis=1)
            for j in label:
                true_labels.append(j)
        
        
        
    elif dataset_name == "MNIST":
        vote_array_path = LOG_DIR_DATA + "/vote_array/MNIST.npy"
        vote_array = np.load(vote_array_path)
        vote_array=vote_array.T
        t,l,r = datasets.get_MNIST_PATE(256)
        
        true_labels = []
        
        for data, label in t:
            for j in label:
                true_labels.append(j)

    elif dataset_name == "FMNIST":
        vote_array_path = LOG_DIR_DATA + "/vote_array/FMNIST.npy"
        vote_array = np.load(vote_array_path)
        vote_array=vote_array.T
        t,l,r = datasets.get_FMNIST_PATE(256)
        
        true_labels = []
        
        for data, label in t:
            for j in label:
                true_labels.append(j)
    
    num_samples = vote_array.shape[0]
    
    predicted_labels = np.zeros(num_samples)
    
    for i in range(len(vote_array)):
        predicted_labels[i] = np.bincount(vote_array[i]).argmax()
    
    
    
    percentage = pate_main.get_how_many_correctly_answered(predicted_labels, true_labels)
    print(percentage)
    
    return percentage



    
