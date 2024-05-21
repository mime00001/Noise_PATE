import datasets
import conventions
import os
from utils import misc

import numpy as np
import torch
import tensorflow as tf

torch.manual_seed(42)

LOG_DIR_DATA = "/disk2/michel/data"

def query_teachers(dataset_name, nb_teachers):
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
        teacher_path = os.path.join(LOG_DIR, dataset_name, teacher_name)
        teacher_nw = torch.load(teacher_path)
        teacher_nw = teacher_nw.to(device)
        teacher_nw.train()
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
    path = LOG_DIR_DATA + "/teacher_labels/{}".format(dataset_name)
    labels = np.array(labels)
    np.save(path, labels, allow_pickle=True)

    
def create_Gaussian_noise(dataset_name, size):
    
    if dataset_name == "MNIST":
        data = np.random.normal(0.0, 1.0, (size, 28, 28, 1))
        path = LOG_DIR_DATA + "/Noise_MNIST/{}".format(size)
        np.save(path, data, allow_pickle=True)
    elif dataset_name == "CIFAR10":
        data = np.random.normal(0.0, 1.0, (size, 32, 32, 3))
        path = LOG_DIR_DATA + "/Noise_CIFAR10/{}".format(size)
        np.save(path, data, allow_pickle=True)
        
    
        
    #TODO finish the function, such that it can accept a dataset shape and then create noise data accordingly
    #TODO CIFAR10 and MNIST noise data 
    
