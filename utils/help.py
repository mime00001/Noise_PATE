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

def run_experiment(dataset_name, n_epochs, nb_teachers=200, threshold=120, sigma_threshold=110, sigma_gnmax=50, epsilon=3, delta=10e-8, num_classes=10):
    
    teachers.util_train_teachers(dataset_name=dataset_name, n_epochs=n_epochs, nb_teachers=nb_teachers)
    
    vote_array = pate_data.query_teachers(dataset_name=dataset_name, nb_teachers=nb_teachers).T
    
    pate_main.inference_pate(vote_array=vote_array, threshold=threshold, sigma_threshold=sigma_threshold, sigma_gnmax=sigma_gnmax, epsilon=epsilon, delta=delta, num_classes=num_classes)
    
    student.util_train_student(dataset_name="MNIST", n_epochs=100, lr=0.001, weight_decay=0, verbose=True, save=True)
    

def remove_rows(input_file, output_file, column_name, condition):
    # Read the CSV file
    df = pd.read_csv(input_file)
    

    # Apply the condition to filter out rows
    filtered_df = df[~df[column_name].apply(condition)]
    

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_file, index=False)


def print_top_values(input_file, column_name, top_n_values):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    #condition = lambda x : x > 5
    
    #df = df[~df["target_epsilon"].apply(condition)]
    
    # Sort the DataFrame by the specified column in descending order
    sorted_df = df.sort_values(by=column_name, ascending=False)
    
    # Get the top N rows
    top_n_df = sorted_df.head(top_n_values)
    
    # Display the top N rows
    print(f"\nTop {top_n_values} rows by {column_name}:")
    print(top_n_df)


def run_parameter_search():
    
    noise_vote_array_path = LOG_DIR_DATA +  "/vote_array/noise_MNIST.npy"

    noise_label_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    
    noise_vote_array = np.load(noise_vote_array_path)
    noise_vote_array=noise_vote_array.T
    
    threshold_list = [80, 150]
    sigma_threshold_list = [50, 80]
    sigma_gnmax_list = [10, 20, 30]
    epsilon_list = [3, 5, 10]
    delta_list =[10e-8]
    num_classes=10
    savepath='./pate_params_new'
    
    
    true_labels=[]
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("noise_MNIST")
    teacher_name = conventions.resolve_teacher_name(experiment_config)
    teacher_path = os.path.join("/disk2/michel", "Pretrained_NW","MNIST", teacher_name)
    teacher_nw = torch.load(teacher_path)
    teacher_nw.to(device)
    teacher_nw.train()
    
    
    t,l,r = datasets.get_noise_MNIST_PATE(256)
    for data, _ in t:
        data = data.to(device)
        with torch.no_grad():
            teacher_output = teacher_nw(data)   
        label = np.argmax(teacher_output.cpu().numpy(), axis=1)
        for j in label:
            true_labels.append(j)
    
    pate_main.tune_pate(noise_vote_array, threshold_list, sigma_threshold_list, sigma_gnmax_list, epsilon_list, delta_list, num_classes, savepath, true_labels)
    
    
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
    
    return valid_acc

def test_ensemble_accuracy():
    
    
    noise_vote_array_path = LOG_DIR_DATA +  "/vote_array/noise_MNIST.npy"

    noise_label_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    
    noise_vote_array = np.load(noise_vote_array_path)
    noise_vote_array=noise_vote_array.T
    
    num_samples = noise_vote_array.shape[0]
    
    predicted_labels = np.zeros(num_samples)
    
    for i in range(len(noise_vote_array)):
        predicted_labels[i] = np.bincount(noise_vote_array[i]).argmax()
    
    true_labels=[]
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("noise_MNIST")
    teacher_name = conventions.resolve_teacher_name(experiment_config)
    teacher_path = os.path.join("/disk2/michel", "Pretrained_NW","MNIST", teacher_name)
    teacher_nw = torch.load(teacher_path)
    teacher_nw.to(device)
    teacher_nw.train()
    
    
    t,l,r = datasets.get_noise_MNIST_PATE(256)
    for data, _ in t:
        data = data.to(device)
        with torch.no_grad():
            teacher_output = teacher_nw(data)   
        label = np.argmax(teacher_output.cpu().numpy(), axis=1)
        for j in label:
            true_labels.append(j)
    
    percentage = pate_main.get_how_many_correctly_answered(predicted_labels, true_labels)
    
    return percentage
    
    
    