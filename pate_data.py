import datasets
import conventions
import os
from utils import misc

import numpy as np
import torch
import tensorflow as tf

torch.manual_seed(42)

LOG_DIR_DATA = "/disk2/michel/data"

def query_teachers(dataset_name : str, nb_teachers : int):
    """queries the teacher ensemble for labels about a specific dataset

    Args:
        dataset_name (str): Name of dataset
        nb_teachers (int): Number of teachers
    """
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset(dataset_name)
    labels = [[] for i in range(nb_teachers)]
    train_loader, _, valid_loader = eval("datasets.get_{}_PATE({})".format( 
            dataset_name,
            experiment_config['batch_size']
    ))
    testdata = next(iter(train_loader))[0].numpy()
    for i in range(nb_teachers):
        print("querying teacher {}".format(i))
        teacher_name = conventions.resolve_teacher_name(experiment_config)
        teacher_name += "_{}".format(i)
        LOG_DIR = '/disk2/michel/Pretrained_NW'
        if dataset_name == "noise_MNIST":
            teacher_path = os.path.join(LOG_DIR, "MNIST", teacher_name)
        elif dataset_name == "noise_CIFAR10":
            teacher_path = os.path.join(LOG_DIR, "CIFAR10", teacher_name)
        else:
            teacher_path = os.path.join(LOG_DIR, dataset_name, teacher_name)
        teacher_nw = torch.load(teacher_path)
        teacher_nw = teacher_nw.to(device)
        
        teacher_nw.train() #set model to training mode, batchnorm trick
        
        testindex = 0
        for data, _ in train_loader:
            if testindex == 0:
                assert np.array_equal(testdata, data.numpy()), "first element is not the same in data, problem with dataloader"
            testindex+=1
            data = data.to(device)
            with torch.no_grad():
                teacher_output = teacher_nw(data)   
            label = np.argmax(teacher_output.cpu().numpy(), axis=1)
            for j in label:
                labels[i].append(j)
    path = LOG_DIR_DATA + "/vote_array/{}".format(dataset_name)
    labels = np.array(labels)
    np.save(path, labels, allow_pickle=True)
    
    return labels

    
def create_Gaussian_noise(dataset_name, size):
    """Creates Gaussian noise, to be fed to the teacher ensemble and used for training the student. Always has mean 0 and std 1.

    Args:
        dataset_name (str): Name of the dataset for which we create noise, either MNIST or CIFAR10
        size (int): Amount of the data to be created, i.e. 10000
    """
    
    np.random.seed(42)
    
    if dataset_name == "MNIST":
        data = np.random.normal(0.0, 1.0, (size, 28, 28))
        path = LOG_DIR_DATA + "/noise_MNIST"
        np.save(path, data, allow_pickle=True)
    elif dataset_name == "CIFAR10":
        data = np.random.normal(0.0, 1.0, (size, 32, 32, 3))
        path = LOG_DIR_DATA + "/noise_CIFAR10"
        np.save(path, data, allow_pickle=True)
        
    

def query_teachers_softmax(dataset_name: str, nb_teachers: int):
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset(dataset_name)
    logits = [[] for i in range(nb_teachers)]
    train_loader, _, valid_loader = eval("datasets.get_{}_PATE({})".format( 
            dataset_name,
            experiment_config['batch_size']
    ))
    testdata = next(iter(train_loader))[0].numpy()
    for i in range(nb_teachers):
        print("querying teacher {}".format(i))
        teacher_name = conventions.resolve_teacher_name(experiment_config)
        teacher_name += "_{}".format(i)
        LOG_DIR = '/disk2/michel/Pretrained_NW'
        if dataset_name == "noise_MNIST":
            teacher_path = os.path.join(LOG_DIR, "MNIST", teacher_name)
        elif dataset_name == "noise_CIFAR10":
            teacher_path = os.path.join(LOG_DIR, "CIFAR10", teacher_name)
        else:
            teacher_path = os.path.join(LOG_DIR, dataset_name, teacher_name)
        teacher_nw = torch.load(teacher_path)
        teacher_nw = teacher_nw.to(device)
        
        teacher_nw.train() #set model to training mode, batchnorm trick
        
        testindex = 0
        for data, _ in train_loader:
            if testindex == 0:
                assert np.array_equal(testdata, data.numpy()), "first element is not the same in data, problem with dataloader"
            testindex+=1
            data = data.to(device)
            with torch.no_grad():
                teacher_output = teacher_nw(data)   
            logit = teacher_output.cpu().numpy()
            for j in logit:
                logits[i].append(j)
    path = LOG_DIR_DATA + "/logit_array/{}".format(dataset_name)
    logits = np.array(logits)
    np.save(path, logits, allow_pickle=True)
    
    return logits



def create_histogram_labels(vote_array):
    
    targets=[]
    
    
    for sample in vote_array:
        unique, counts = np.unique(sample, return_counts=True)
        count_array = np.zeros(10, dtype=int)
        count_array[unique]=counts
        targets.append(count_array)
    
    return targets
        



def create_logit_labels(logit_array):
    
    num_samples = logit_array.shape[0]
    
    targets = []
    
    for sample in logit_array:
        combined_logits = sample[0]
        for logit in range(1, len(sample)):
            combined_logits += logit
        combined_logits /= len(sample)
        targets.append(combined_logits)
    
    return targets
        
    
    