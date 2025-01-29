import torch
import os
import numpy as np
import pate_main

import conventions
from utils import misc
import random
from utils import teachers

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

LOG_DIR_DATA = "/data"
LOG_DIR = ""
#get the greedy teacher accuracy

def query_one_less_teacher(teacher_id, nb_teachers, query_dataset, target_dataset, BN_trick, SSL):
    
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
    labels = [[] for i in range(nb_teachers-1)]
    
    
    #need to adjust the dataloaders
    
    
    train_loader, _, valid_loader =get_capc_dataloaders()
    
    
    testdata = next(iter(train_loader))[0].numpy()
    for i in range(nb_teachers):
        if i != teacher_id:
            print("querying teacher {}".format(i))
            teacher_name = "teacher_MNIST_mnistresnet.model"
            teacher_name += "_{}".format(i)
            LOG_DIR = f'/Pretrained_NW/SL_'
            
            if "noise_" in target_dataset:
                target_dataset = target_dataset.replace("noise_", "")
            
            teacher_path = os.path.join("/Pretrained_NW/SL_MNIST", teacher_name)
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
                    if i > teacher_id:
                        labels[i-1].append(j)
                    else:
                        labels[i].append(j)
    path = LOG_DIR_DATA + "/vote_array/capc/{}".format(query_dataset)
    labels = np.array(labels)
    print(f"Saving votes at {path}.")
    np.save(path, labels, allow_pickle=True)
    
    return labels
    
    
def get_capc_dataloaders():
    batch_size = 128
    num_workers = 4
    validation_size = 0.1
    

    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

     #, transform=transform_train
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    end = int(len(testset)*(1-validation_size))
    
    partition_train = [testset[i] for i in range(end)]
    partition_test = [testset[i] for i in range(end, len(testset))]
    
    train_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, valid_loader, test_loader


    
def get_greedy_dataloaders(teacher_id, nb_teachers):
    
    transform_train = transform=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])


    trainset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train

    batch_len = int(len(trainset) / nb_teachers)
    
    start = teacher_id * batch_len
    end = (teacher_id+1) * batch_len
        
    old_partition_train = [trainset[i] for i in range(start, end)]
    
    
    
    batch_size = 128
    num_workers = 4
    validation_size = 0.1

    transform_test = transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)

    end = int(len(testset)*(1-validation_size))
    
    target_path = LOG_DIR_DATA + "/teacher_labels/capc/MNIST.npy"
    
    teacher_labels = np.load(target_path)
    
    greedy_partition_train = [[testset[i][0], torch.tensor(teacher_labels[i])] for i in range(end) if teacher_labels[i]!= -1] #remove all datapoints, where we have no answer from the teacher ensemble
    partition_test = [testset[i] for i in range(end, len(testset))]

    accuracy= 0
    for i in range(end):
        if testset[i][1] == teacher_labels[i]:
            accuracy +=1 

    accuracy = accuracy/len(greedy_partition_train)
    
    augmented_partition_train = ConcatDataset([greedy_partition_train, old_partition_train])
    
    print(f"Num samples for the greedy teacher: {len(greedy_partition_train)}")
    print(f"Accuracy for the greedy teacher: {accuracy}")
    
    train_loader = torch.utils.data.DataLoader(augmented_partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, valid_loader, test_loader

def get_fair_dataloaders(teacher_id, nb_teachers):
    transform_train = transform=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])


    trainset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train

    batch_len = int(len(trainset) / nb_teachers)
    
    start = teacher_id * batch_len
    end = (teacher_id+1) * batch_len
        
    old_partition_train = [trainset[i] for i in range(start, end)]
    
    
    batch_size = 128
    num_workers = 4
    validation_size = 0.1

    transform_test = transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    end = int(len(testset)*(1-validation_size))
    
    target_path = LOG_DIR_DATA + "/teacher_labels/capc/MNIST.npy"
    
    teacher_labels = np.load(target_path)
    
    partition_train = [[testset[i][0], torch.tensor(teacher_labels[i])] for i in range(end) if teacher_labels[i]!= -1] #remove all datapoints, where we have no answer from the teacher ensemble
    partition_test = [testset[i] for i in range(end, len(testset))]

    all_num_samples = len(partition_train)
    
    fair_num_samples = all_num_samples//200
    
    fair_partition_train =  random.sample(partition_train, fair_num_samples)
    
    accuracy= 0
    for i in range(end):

        if testset[i][1] == teacher_labels[i]:
            accuracy +=1 

    accuracy = accuracy/len(partition_train)
    
    augmented_partition_train = ConcatDataset([fair_partition_train, old_partition_train])
    
    print(f"Num samples for the fair teacher: {len(augmented_partition_train)}")
    print(f"Accuracy for the fair teacher: {accuracy}")
    
    train_loader = torch.utils.data.DataLoader(augmented_partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, valid_loader, test_loader

def greedy_teacher_and_fair_teacher(nb_teachers, query_dataset, target_dataset, SSL):
    #first greedy teacher is excluded and asks all the queries to the rest
    #then teacher is finetuned on the answered data and labels
    device = "cuda"
    experiment_config = conventions.resolve_dataset(target_dataset)
    teacher_id = random.randint(0, 199)
    params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 5, "delta" : 1e-5}
    
    noise_vote_array = query_one_less_teacher(nb_teachers=nb_teachers, teacher_id=teacher_id, query_dataset=query_dataset, target_dataset=target_dataset, BN_trick=False, SSL=False)
    noise_vote_array = noise_vote_array.T
    
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/capc/{}.npy".format(query_dataset)
    eps, noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
    num_answered = (noise_votes != -1).sum()
    
    train_loader, _, test_loader = get_greedy_dataloaders()
    
    teacher_name = "teacher_MNIST_mnistresnet.model"
    teacher_name += "_{}".format(teacher_id)
    
    teacher_path = os.path.join(LOG_DIR + "Pretrained_NW/SL_MNIST", teacher_name)
    
    teacher_nw = torch.load(teacher_path)
    teacher_nw = teacher_nw.to(device)
    #lr=1e-3, weight_decay=0, verbose=True, save=True, LOG_DIR='/storage3/michel/',
    #(teacher_nw, teacher_id, nb_teachers, train_loader, valid_loader, n_epochs, lr, weight_decay, verbose, device, save, LOG_DIR)
    # train greedy teacher
    metrics = teachers.train_teacher(teacher_nw=teacher_nw, train_loader=train_loader, valid_loader=test_loader, n_epochs=10, teacher_id=teacher_id, lr=1e-5, 
                                     weight_decay=0, verbose=True, save=False, LOG_DIR='/storage3/michel/', nb_teachers=nb_teachers, device="cuda")
    print(metrics)
    
    greedy_teacher_acc = metrics[3][-1]
    
    train_loader, _, test_loader = get_fair_dataloaders()
    
    teacher_name = "teacher_MNIST_mnistresnet.model"
    teacher_name += "_{}".format(teacher_id)
    
    teacher_path = os.path.join(LOG_DIR + "Pretrained_NW/SL_MNIST", teacher_name)
    teacher_nw = torch.load(teacher_path)
    teacher_nw = teacher_nw.to(device)
    
    # train fair teacher
    metrics = teachers.train_teacher(teacher_nw=teacher_nw, train_loader=train_loader, valid_loader=test_loader, n_epochs=10, teacher_id=teacher_id, lr=1e-5,
                                    weight_decay=0, verbose=True, save=False, LOG_DIR='',nb_teachers=nb_teachers, device="cuda")
    print(metrics)
    
    fair_teacher_acc = metrics[3][-1]
    
    print(f"Teacher being left out: {teacher_id}")
    print(f"Greedy teacher accuracy: {greedy_teacher_acc}\t Fair teacher accuracy: {fair_teacher_acc}")
    
    return greedy_teacher_acc, fair_teacher_acc
    
    #then fair teacher, i.e. split the data randomly by 200 and train teacher again on the data
    
    
def avg_teacher_accs():
    
    greedy_accs = []
    
    fair_accs = []
    
    for i in range(10):
        g, f = greedy_teacher_and_fair_teacher(200, "MNIST", "MNIST", SSL=False)
        greedy_accs.append(g)
        fair_accs.append(f)
    
    
    
    print(f"Average greedy acc: {np.mean(greedy_accs)}\t Average fair acc: {np.mean(fair_accs)}")
    print(f"Std greedy acc: {np.std(greedy_accs)}\t Std fair acc: {np.std(fair_accs)}")