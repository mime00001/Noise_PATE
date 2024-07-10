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

import matplotlib.pyplot as plt
from PIL import Image
import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
import os
from utils import misc

import pandas as pd


LOG_DIR_DATA = "/storage3/michel/data"
LOG_DIR = "/storage3/michel/"


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


def run_parameter_search(path="/vote_array/noise_SVHN.npy", savepath="./pate_params", predicted_labels=None):
    
    noise_vote_array_path = LOG_DIR_DATA +  path
    
    noise_vote_array = np.load(noise_vote_array_path)
    noise_vote_array=noise_vote_array.T
    
    threshold_list = [150, 250, 280, 300] #
    sigma_threshold_list = [100, 150, 200] # 
    sigma_gnmax_list = [20, 40, 50]
    epsilon_list = [20]
    delta_list =[1e-6]
    num_classes=10
    
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


def count_answered_labels(teacher_label_path=None):
    
    if not teacher_label_path:
        teacher_label_path = LOG_DIR_DATA + "/teacher_labels/MNIST.npy"
    
    teacher_labels = np.load(teacher_label_path)
    
    mask = teacher_labels != -1
    
    count = mask.sum()
    print(count)


def show_images(num=5, padding=1):
    teacher_label_path = LOG_DIR_DATA + "/teacher_labels/noise_SVHN.npy"
    data_path = LOG_DIR_DATA + "/noise_SVHN.npy"
    
    teacher_labels = np.load(teacher_label_path)
    data = np.load(data_path)
    val=0
    
    labels=[]
    
    images=[]
    for i, im in enumerate(data):   
         
        if val==num:
            break
        if teacher_labels[i] != -1:
            min_val = im.min(axis=(1, 2), keepdims=True) #axis=(0, 1), keepdims=True
            max_val = im.max(axis=(1, 2), keepdims=True)
            normalized_array = ((im - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
            new_image = Image.fromarray(normalized_array.transpose(1, 2, 0), mode="RGB")
            save_path = LOG_DIR + "/Images/SVHN"+ str(i)+".jpeg"
            
            labels.append(teacher_labels[i])
            
            new_image.save(save_path)
            images.append(new_image)
            val+=1
            
    num_rows = 2
    num_cols = (num + 1) // 2
    width, height = images[0].size
    print(images[0].size)
    total_width = num_cols * width + (num_cols - 1) * padding
    total_height = num_rows * height + (num_rows - 1) * padding
    
    grid_image = Image.new('RGB', size=(total_width, total_height), color=(255, 255, 255))
    
    for idx, img in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        x = col * (width + padding)
        y = row * (height + padding)
        grid_image.paste(img, (x, y))
    
    save_path = os.path.join(LOG_DIR, "Images", "SVHN_grid.jpeg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grid_image.save(save_path)
    print(labels)
    
def consensus_calculations(path="/disk2/michel/Plots/consensus_diff_noiseMNIST.npy"):
    arr = np.load(path)
    
    print(f"mean: {np.mean(arr)}")
    print(f"meadian: {np.median(arr)}")
    print(f"max: {np.amax(arr)}")
    print(f"min: {np.amin(arr)}")
    print(f"std: {np.std(arr)}")
    
    
    
def table1_help():
    
    np.set_printoptions(suppress=True)
    
    public_list= [[(5.0, 0.95, 2989), (5.0, 0.967, 2983), (5.0, 0.961, 2936), (5.0, 0.96, 2897), (5.0, 0.961, 2906)], [(6.476, 0.972, 4712), (6.476, 0.961, 4756), (6.476, 0.968, 4700), (6.476, 0.962, 4670), (6.476, 0.958, 4647)], [(6.476, 0.959, 4699), (6.476, 0.97, 4670), (6.476, 0.961, 4705), (6.476, 0.968, 4652), (6.476, 0.965, 4659)], [(6.476, 0.962, 4703), (6.476, 0.968, 4653), (6.476, 0.959, 4761), (6.476, 0.966, 4666), (6.476, 0.966, 4734)]]
    gaussian_list = [[(5.0, 0.354, 1228), (5.0, 0.374, 1226), (5.0, 0.419, 1208), (5.0, 0.398, 1256), (5.0, 0.349, 1184)], [(8.0, 0.524, 2698), (8.0, 0.478, 2662), (8.0, 0.506, 2681), (8.0, 0.515, 2708), (8.0, 0.523, 2746)], [(10.0, 0.515, 3874), (10.0, 0.549, 3842), (10.0, 0.557, 3913), (10.0, 0.566, 4087), (10.0, 0.586, 4041)], [(20.0, 0.765, 11592), (20.0, 0.747, 11833), (20.0, 0.718, 11804), (20.0, 0.777, 11834), (20.0, 0.724, 11748)]]
    FMNIST_list = [[(5.0, 0.461, 1774), (5.0, 0.452, 1762), (5.0, 0.505, 1742), (5.0, 0.456, 1782), (5.0, 0.483, 1801)], [(8.0, 0.525, 4039), (8.0, 0.547, 4068), (8.0, 0.535, 4003), (8.0, 0.527, 3990), (8.0, 0.568, 3989)], [(10.0, 0.61, 5821), (10.0, 0.579, 5843), (10.0, 0.607, 5800), (10.0, 0.63, 5837), (10.0, 0.57, 5866)], [(20.0, 0.678, 17734), (20.0, 0.672, 17573), (20.0, 0.685, 17505), (20.0, 0.658, 17586), (20.0, 0.686, 17620)]]
    
    headers = ['eps=5', 'eps=8', "eps=10", "eps=20"]
    row_labels = [ "public_data", "Gaussian noise", "FMNIST data"]
    values = [
        [(np.mean(public_list[0], axis=0), np.std(public_list[0], axis=0)), (np.mean(public_list[1], axis=0), np.std(public_list[1], axis=0)), (np.mean(public_list[2], axis=0), np.std(public_list[2], axis=0)), (np.mean(public_list[3], axis=0), np.std(public_list[3], axis=0))],
        [(np.mean(gaussian_list[0], axis=0), np.std(gaussian_list[0], axis=0)), (np.mean(gaussian_list[1], axis=0), np.std(gaussian_list[1], axis=0)), (np.mean(gaussian_list[2], axis=0), np.std(gaussian_list[2], axis=0)), (np.mean(gaussian_list[3], axis=0), np.std(gaussian_list[3], axis=0))],
        [(np.mean(FMNIST_list[0], axis=0), np.std(FMNIST_list[0], axis=0)), (np.mean(FMNIST_list[1], axis=0), np.std(FMNIST_list[1], axis=0)), (np.mean(FMNIST_list[2], axis=0), np.std(FMNIST_list[2], axis=0)), (np.mean(FMNIST_list[3], axis=0), np.std(FMNIST_list[3], axis=0))]
    ]
    
    cellText1 = [[] for i in range(3)]
    cellText2 = [[] for i in range(3)]
    
    for i, v in enumerate(values):
        for mean_std in v:
            m, s = mean_std
            #m, s = np.asarray(m), np.asarray(s)
            cellText1[i].append(f"{m.round(3)}")
            cellText2[i].append(f"{s.round(3)}")
        
            
    fig, ax = plt.subplots(figsize=(20, 10))

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=cellText1, colLabels=headers, rowLabels=row_labels, loc='center', cellLoc='center')
    table.set_fontsize(20)

    # Adjust layout
    plt.subplots_adjust(left=0.2, top=0.8)

    # Save the table to a file
    plt.savefig('table 1_mean.png')
    
    
    fig, ax = plt.subplots()

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=cellText2, colLabels=headers, rowLabels=row_labels, loc='center', cellLoc='center')
    table.set_fontsize(16)

    # Adjust layout
    plt.subplots_adjust(left=0.2, top=0.8)

    # Save the table to a file
    plt.savefig('table 1_std.png')
