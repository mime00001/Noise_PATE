import datasets
import conventions
import os
from utils import misc

import numpy as np
import torch
import tensorflow as tf

LOG_DIR_DATA = "data"
LOG_DIR = ""
LOG_DIR_MODEL = ""
#this file performs the data processing

def query_teachers(target_dataset : str, query_dataset :str, nb_teachers : int, BN_trick=True, SSL=True):
    """queries the teacher ensemble for labels about a specific dataset

    Args:
        dataset_name (str): Name of dataset
        nb_teachers (int): Number of teachers
    """
    if SSL:
        prefix=""
    else:
        prefix="SL_"
    
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset(target_dataset)
    labels = [[] for i in range(nb_teachers)]
    train_loader, _, valid_loader = eval("datasets.get_{}_PATE({})".format( 
            query_dataset,
            experiment_config['batch_size']
    ))
    testdata = next(iter(train_loader))[0].numpy()
    for i in range(nb_teachers):
        print("querying teacher {}".format(i))
        teacher_name = conventions.resolve_teacher_name(experiment_config)
        teacher_name += "_{}".format(i)
        
        if "noise_" in target_dataset:
            target_dataset = target_dataset.replace("noise_", "")
        
        teacher_path = os.path.join(LOG_DIR, f"{prefix}Pretrained_NW", target_dataset, teacher_name)
        teacher_nw = torch.load(teacher_path)
        teacher_nw = teacher_nw.to(device)
        
        if experiment_config['BN_trick'] and BN_trick:
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
    path = LOG_DIR_DATA + "/vote_array/{}".format(query_dataset)
    labels = np.array(labels)
    print(f"Saving votes at {path}.")
    np.save(path, labels, allow_pickle=True)
    
    return labels

    
def create_Gaussian_noise(dataset_name, size):
    """Creates Gaussian noise, to be fed to the teacher ensemble and used for training the student. Always has mean 0 and std 1.

    Args:
        dataset_name (str): Name of the dataset for which we create noise, either MNIST or CIFAR10
        size (int): Amount of the data to be created, e.g. 10000
    """
    
    np.random.seed(42)
    
    if dataset_name == "MNIST":
        data = np.random.normal(0.0, 1.0, (size, 28, 28))
        path = LOG_DIR_DATA + "/noise_MNIST"
        np.save(path, data, allow_pickle=True)
    elif dataset_name == "CIFAR10":
        data = np.random.normal(0.0, 1.0, (size, 3, 32, 32))
        path = LOG_DIR_DATA + "/noise_CIFAR10"
        np.save(path, data, allow_pickle=True)
        
    
