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
from utils import misc

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
    
    pate_main.inference_pate(vote_array=noise_vote_array, threshold=120, sigma_threshold=110, sigma_gnmax=50, epsilon=10, delta=10e-8, num_classes=10, savepath=noise_label_path) 
    
    noise_votes = np.load(noise_label_path)
    
    
    
    unique_v, counts = np.unique(noise_votes, return_counts=True)
    
    for value, count in zip(unique_v, counts):
        print(f"Value {value} occurs {count} times")
    
    student.util_train_student(dataset_name="noise_MNIST", n_epochs=50, lr=0.001, optimizer="Adam") 
    
    
    
    
    
    
    
    



#ne()

#for mnist epochs = 35, lr = 0.001, decay = 0, sigma1 = 110, sigma2 = 50, delta = 10e-8, epsilon =3, T=120


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
    
    condition = lambda x : x > 5
    
    df = df[~df["target_epsilon"].apply(condition)]
    
    # Sort the DataFrame by the specified column in descending order
    sorted_df = df.sort_values(by=column_name, ascending=False)
    
    # Get the top N rows
    top_n_df = sorted_df.head(top_n_values)
    
    # Display the top N rows
    print(f"\nTop {top_n_values} rows by {column_name}:")
    print(top_n_df)


# Example usage
# Define the condition as a lambda function
condition = lambda x: x < 50  # Example: remove rows where the value in the column is greater than 10

# Specify the input and output file paths, and the column to apply the condition on
input_file = 'pate_params_2.csv'
output_file = 'pate_params_2_rf.csv'
column_name = 'num_correctly_answered'  # Replace with your actual column name

#print_top_values(input_file, column_name, 20)
#remove_rows(input_file, output_file, column_name, condition)
