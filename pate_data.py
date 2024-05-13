import datasets
import conventions
import os
from utils import misc

import torch
import tensorflow as tf

torch.manual_seed(42)

def query_teachers(dataset_name, teacher_path, nb_teachers):
    
    labels = []
    
    for i in range(nb_teachers):
        experiment_config = conventions.resolve_dataset(dataset_name)
        teacher_name = conventions.resolve_teacher_name(experiment_config)
        LOG_DIR = '/disk2/michel/Pretrained_NW'
        teacher_path = os.path.join(LOG_DIR, teacher_name, "_{}".format(i))
        teacher_nw = torch.load(teacher_path)
        train_loader, _, valid_loader = eval("datasets.get_{}({}, {}, {})".format(
            dataset_name,
            experiment_config['batch_size'],
            i,
            nb_teachers
        ))
        for data, _ in train_loader:
            label = teacher_nw(data)
            labels.append(label)
    #TODO save    

    
def create_Gaussian_noise(dataset_name, size, batchsize):
    train_loader, _, valid_loader = eval("datasets.get_{}({}, {}, {})".format(
        dataset_name,
        batchsize,
        1,
        100
    ))
    device = misc.get_device()
    
    data_sample = next(iter(valid_loader))[0]
    amount = size//batchsize
    for i in range(amount):
        data = torch.randn_like(data_sample, device=device)
        
    #TODO finish the function, such that it can accept a dataset shape and then create noise data accordingly
    #TODO CIFAR10 and MNIST noise data 
    
