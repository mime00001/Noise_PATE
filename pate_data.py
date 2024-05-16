import datasets
import conventions
import os
from utils import misc

import numpy as np
import torch
import tensorflow as tf

torch.manual_seed(42)

def query_teachers(dataset_name, teacher_path, nb_teachers):
    device = misc.get_device()
    
    labels = []
    
    for i in range(nb_teachers):
        experiment_config = conventions.resolve_dataset(dataset_name)
        teacher_name = conventions.resolve_teacher_name(experiment_config)
        LOG_DIR = '/disk2/michel/Pretrained_NW'
        teacher_path = os.path.join(LOG_DIR, dataset_name, teacher_name, "_{}".format(i))
        teacher_nw = torch.load(teacher_path)
        teacher_nw.to(device)
        teacher_nw.train()
        train_loader, _, valid_loader = eval("datasets.get_{}_PATE({})".format( #might have to redo this
            dataset_name,
            experiment_config['batch_size']
        ))
        for data, _ in train_loader:
            with torch.no_grad():
                teacher_output = teacher_nw(data)
                
            label = np.argmax(teacher_output, axis=1)
            labels.append(label)
            
        
    #TODO save    

    
def create_Gaussian_noise(dataset_name, size):
    train_loader, _, valid_loader = eval("datasets.get_{}({}, {}, {})".format(
        dataset_name,
        256,
        1,
        100
    ))
    device = misc.get_device()
    batchsize = 256    
    data_sample = next(iter(valid_loader))[0]
    amount = size//batchsize + (size % batchsize > 0) 
    for i in range(amount):
        data = torch.randn_like(data_sample, device=device)
        
    #TODO finish the function, such that it can accept a dataset shape and then create noise data accordingly
    #TODO CIFAR10 and MNIST noise data 
    
