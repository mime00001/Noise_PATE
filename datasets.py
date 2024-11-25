"""
Scripts to get dataloaders for
MNIST, CIFAR10, etc.
"""
import torch.utils
import torch.utils.data
from utils import util
from PIL import Image, ImageOps
import os

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

LOG_DIR_DATA = "/storage3/michel/data"
LOG_DIR = "/storage3/michel"
LOG_DIR_MODEL = "/storage3/michel"

def getDataloaders(trainset, testset, valid_size, batch_size, num_workers):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader, valid_loader, test_loader


################################################################
# Datasets
################################################################

#this code is partially taken from https://github.com/Piyush-555/GaussianDistillation/tree/main

#these datasets are for teacher training, they return three dataloaders, where the first data loader can be used as training data loader and the third as validation loader

def get_CIFAR10(batch_size, teacher_id, nb_teachers, valid_size=0.2):
    num_workers = 4
    
    assert int(teacher_id) < int(nb_teachers)

    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49139969, 0.48215842, 0.44653093), (0.24703223,0.24348513, 0.26158784)), #(0.2023, 0.1994, 0.2010)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49421429, 0.4851314, 0.45040911), (0.24665252, 0.24289226, 0.26159238)),
    ])
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.CIFAR10(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test) #, transform=transform_test
    batch_len = int(len(trainset) / nb_teachers)
    assert batch_len >= batch_size, f"len trainset {len(trainset)} and batchlen {batch_len}"
    
    start = teacher_id * batch_len
    end = (teacher_id+1) * batch_len
        
    partition_train = [trainset[i] for i in range(start, end)]

    return getDataloaders(partition_train, testset, 0.0, batch_size, num_workers)  # train_loader, valid_loader, test_loader

def get_MNIST(batch_size, teacher_id, nb_teachers, valid_size=0.2):
    num_workers = 4
    
    assert int(teacher_id) < int(nb_teachers)

    transform_train = transform=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

    trainset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test) #, transform=transform_test
    batch_len = int(len(trainset) / nb_teachers)
    assert batch_len >= batch_size, "batchsize  too large for number of teachers, each teacher has less than batchsize samples"
    
    start = teacher_id * batch_len
    end = (teacher_id+1) * batch_len
        
    partition_train = [trainset[i] for i in range(start, end)]

    return getDataloaders(partition_train, testset, 0.0, batch_size, num_workers)  # train_loader, valid_loader, test_loader

def get_FMNIST(batch_size, teacher_id, nb_teachers, valid_size=0.2):
    num_workers = 4
    
    assert int(teacher_id) < int(nb_teachers)

    transform_train=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.2860,), (0.3530,)) # normalize inputs
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.2860,), (0.3530,)) # normalize inputs
    ])

    trainset = torchvision.datasets.FashionMNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.FashionMNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    batch_len = int(len(trainset) / nb_teachers)
    assert batch_len >= batch_size, "batchsize  too large for number of teachers, each teacher has less than batchsize samples"
    
    start = teacher_id * batch_len
    end = (teacher_id+1) * batch_len
        
    partition_train = [trainset[i] for i in range(start, end)]

    return getDataloaders(partition_train, testset, 0.0, batch_size, num_workers)  # train_loader, valid_loader, test_loader

def get_SVHN(batch_size, teacher_id, nb_teachers, valid_size=0.2):
    num_workers = 4
    
    assert int(teacher_id) < int(nb_teachers)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.44921386, 0.4496643, 0.45029628), (0.20032172, 0.19916263, 0.19936596)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.45207793, 0.45359373, 0.45602703), (0.22993235, 0.229334, 0.2311905)),
    ])

    transform_extra =transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.42997558, 0.4283771, 0.44269393), (0.19630221, 0.1978732, 0.19947216))
    ])

    trainset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="train", download=True, transform=transform_train) #, transform=transform_train
    
    extraset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="extra", download=True, transform=transform_extra)
    
    testset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="test", download=True, transform=transform_test) #, transform=transform_test

    trainset= ConcatDataset([trainset, extraset])

    

    batch_len = int(len(trainset) / nb_teachers)
    assert batch_len >= batch_size, f"length of trainset {len(trainset)}, batch len: {batch_len}"
    
    start = teacher_id * batch_len
    end = (teacher_id+1) * batch_len
        
    partition_train = [trainset[i] for i in range(start, end)]

    return getDataloaders(partition_train, testset, 0.0, batch_size, num_workers)


#these datasets are for querying the teachers, they return three dataloaders,
# where only the first one should be used for teacher querying, the last one is the validation set for the student

def get_CIFAR10_PATE(batch_size, validation_size=0.2):
    
    num_workers = 4

    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49139969, 0.48215842, 0.44653093), (0.24703223,0.24348513, 0.26158784)), #(0.2023, 0.1994, 0.2010)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49421429, 0.4851314, 0.45040911), (0.24665252, 0.24289226, 0.26159238)),
    ])
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.CIFAR10(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test) #, transform=transform_test
    
    #end = int(len(trainset)*(1-validation_size))
    end = len(testset)
    
    partition_train = [testset[i] for i in range(end)]
    partition_test = [testset[i] for i in range(end, len(testset))]
    
    train_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, valid_loader, test_loader

def get_noise_CIFAR10_PATE(batch_size):
    num_workers = 4
    
    path = LOG_DIR_DATA + "/noise_CIFAR10.npy"
    
    data_set = np.load(path)
    
    train_data = [(torch.FloatTensor(data_set[i]), torch.tensor(0)) for i in range(len(data_set))] #will probably need to rewrite this, dont know if unsqueeze is necessary for CIFAR10 noise data
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, train_loader, train_loader #return same dataloader so i dont have to rewrite function

def get_MNIST_PATE(batch_size, validation_size=0.2):
    num_workers = 4
    transform_train = transform=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

    trainset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    end = int(len(testset)*(1-validation_size))
    
    partition_train = [testset[i] for i in range(end)]
    partition_test = [testset[i] for i in range(end, len(testset))]
    
    train_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, valid_loader, test_loader

def get_noise_MNIST_PATE(batch_size):
    num_workers = 4
    
    path = LOG_DIR_DATA + "/noise_MNIST.npy"
    
    data_set = np.load(path)
    
    train_data = [(torch.FloatTensor(data_set[i]).unsqueeze(0), torch.tensor(0)) for i in range(len(data_set))]
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, train_loader, train_loader #return same dataloader so i dont have to rewrite function

def get_FMNIST_PATE(batch_size, validation_size=0.2):
    num_workers = 4
    transform_train = transform=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.2860,), (0.3530,)) # normalize inputs
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.2860,), (0.3530,)) # normalize inputs
    ])

    trainset = torchvision.datasets.FashionMNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.FashionMNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    """ end = int(len(testset)*(1-validation_size))
    
    partition_train = [testset[i] for i in range(end)]
    partition_test = [testset[i] for i in range(end, len(testset))]
     """
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, valid_loader, test_loader

def get_SVHN_PATE(batch_size, validation_size=0.2):
    num_workers = 4
    

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.44921386, 0.4496643, 0.45029628), (0.20032172, 0.19916263, 0.19936596)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.45207793, 0.45359373, 0.45602703), (0.22993235, 0.229334, 0.2311905)),
    ])

    trainset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="train", download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="test", download=True, transform=transform_test) #, transform=transform_test

    end = int(len(testset)*(1-validation_size))
    
    partition_train = [testset[i] for i in range(end)]
    partition_test = [testset[i] for i in range(end, len(testset))]

    train_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, valid_loader, test_loader

def get_noise_SVHN_PATE(batch_size):
    num_workers = 4
    
    path = LOG_DIR_DATA + "/noise_SVHN.npy"
    
    data_set = np.load(path)
    
    train_data = [(torch.FloatTensor(data_set[i]), torch.tensor(0)) for i in range(len(data_set))]
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, train_loader, train_loader #return same dataloader so i dont have to rewrite function

def get_dead_leaves_PATE(batch_size):
    
    num_workers = 4
    
    path = LOG_DIR_DATA + "/dead_leaves-mixed.npy"
    
    images= np.load(path)
    
    mean = images.mean()
    std = images.std()
    
    train_data = [(torch.FloatTensor((images[i]- mean)/std).unsqueeze(0), torch.tensor(0)) for i in range(len(images))]
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, train_loader, train_loader #return same dataloader so i dont have to rewrite function

def get_stylegan_PATE(batch_size):
    
    """ path = LOG_DIR_DATA + "/stylegan-oriented/"
    
    images=[]
    
    for image in os.listdir(path):
        images.append(ImageOps.grayscale(Image.open((path + image))).resize((28, 28)))
        
    #need to be normalized before putting into network
    images = np.array(images) """
    
    
    
    num_workers=4
    
    path = LOG_DIR_DATA + "/stylegan-oriented.npy"
    
    images = np.load(path)
    
    mean = images.mean()
    std = images.std()
    
    train_data = [(torch.FloatTensor((images[i]- mean)/std).unsqueeze(0), torch.tensor(0)) for i in range(len(images))]
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, train_loader, train_loader

def get_SVHN_MNIST_PATE(batch_size):
    
    num_workers = 4

    trainset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="train", download=True) #, transform=transform_train
    testset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="test", download=True) #, transform=transform_test

    images = [Image.fromarray(np.transpose(image, (1, 2, 0))) for image in trainset.data]
    
    gray_images = [ImageOps.grayscale(image.resize((28, 28))) for image in images]
    
    gray_images = np.array(gray_images)
    mean = gray_images.mean()
    std = gray_images.std()
    
    trainset = [(torch.FloatTensor((gray_images[i]- mean)/std).unsqueeze(0), torch.tensor(0)) for i in range(len(gray_images))]
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
   
    return train_loader, train_loader, train_loader
    
def get_FractalDB_PATE(batch_size):
    
    """ path = LOG_DIR_DATA + "/FractalDB/"
    
    images=[]
    
    for image in os.listdir(path):
        images.append(ImageOps.grayscale(Image.open((path + image))).resize((28, 28)))
        
    #need to be normalized before putting into network
    images = np.array(images) 
    path = LOG_DIR_DATA + "/FractalDB.npy"
    np.save(path, images) """
    
    
    num_workers=4
    
    path = LOG_DIR_DATA + "/FractalDB.npy"
    
    images = np.load(path)
    
    mean = images.mean()
    std = images.std()
    
    train_data = [(torch.FloatTensor((images[i]- mean)/std).unsqueeze(0), torch.tensor(0)) for i in range(len(images))]
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, train_loader, train_loader

def get_Shaders21k_PATE(batch_size):
    """ path = LOG_DIR_DATA + "/shaders21k/"
    
    images=[]
    for image in os.listdir(path):
        images.append(ImageOps.grayscale(Image.open((path + image))).resize((28, 28)))
        
    #need to be normalized before putting into network
    images = np.array(images) 
    
    
    path = LOG_DIR_DATA + "/Shaders21k.npy"
    np.save(path, images) """
    
    
    num_workers=4
    
    path = LOG_DIR_DATA + "/Shaders21k.npy"
    
    images = np.load(path)
    
    mean = images.mean()
    std = images.std()
    
    num_points = 100000
    if len(images) < num_points:
        num_points = len(images)
    
    train_data = [(torch.FloatTensor((images[i]- mean)/std).unsqueeze(0), torch.tensor(0)) for i in range(num_points)]
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, train_loader, train_loader

def get_MIX_PATE(batch_size):
    num_workers = 4
    
    samples_per_dataset = 5000
    
    # Load some Gaussian noise
    noise_path = LOG_DIR_DATA + "/noise_MNIST.npy"
    noise_data = np.load(noise_path)
    
    noise_train_data = [(torch.FloatTensor(noise_data[i]).unsqueeze(0), torch.tensor(0)) for i in range(samples_per_dataset)]    
    # Load some FMNIST
    transform_train = transform=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.2860,), (0.3530,)) # normalize inputs
    ])

    fmnist_trainset = torchvision.datasets.FashionMNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train)
    
    fmnist_train_data = [(fmnist_trainset[i][0], torch.tensor(0)) for i in range(samples_per_dataset)]
    
    
    # Load some dead leaves
    leaves_path = LOG_DIR_DATA + "/dead_leaves-mixed.npy"
    
    leaves= np.load(leaves_path)
    
    mean = leaves.mean()
    std = leaves.std()
    
    leaves_train_data = [(torch.FloatTensor((leaves[i]- mean)/std).unsqueeze(0), torch.tensor(0)) for i in range(samples_per_dataset)]
    
    # Load some StyleGAN
    stylegan_path = LOG_DIR_DATA + "/stylegan-oriented.npy"
    
    stylegan = np.load(stylegan_path)
    
    mean = stylegan.mean()
    std = stylegan.std()
    
    stylegan_train_data = [(torch.FloatTensor((stylegan[i]- mean)/std).unsqueeze(0), torch.tensor(0)) for i in range(samples_per_dataset)]
    # Load some FractalDB
    
    fractaldb_path = LOG_DIR_DATA + "/FractalDB.npy"
    
    fractaldb = np.load(fractaldb_path)
    
    mean = fractaldb.mean()
    std = fractaldb.std()
    
    fractaldb_train_data = [(torch.FloatTensor((fractaldb[i]- mean)/std).unsqueeze(0), torch.tensor(0)) for i in range(samples_per_dataset)]
    
    # Load some Shaders21k  
    shaders_path = LOG_DIR_DATA + "/Shaders21k.npy"
    
    shaders = np.load(shaders_path)
    
    mean = shaders.mean()
    std = shaders.std()
    
    shaders_train_data = [(torch.FloatTensor((shaders[i]- mean)/std).unsqueeze(0), torch.tensor(0)) for i in range(samples_per_dataset)]
    
    train_data = noise_train_data + fmnist_train_data + leaves_train_data + stylegan_train_data + fractaldb_train_data + shaders_train_data
    
    traindata_data = [data[0] for data in train_data]
    np.save(LOG_DIR_DATA + "/MIX.npy", traindata_data)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, train_loader, train_loader
        
def get_dead_leaves_SVHN_PATE(batch_size):
    num_workers=4
    
    """ path = LOG_DIR_DATA + "/dead_leaves-mixed/"
    
    images=[]
    for image in os.listdir(path):
        images.append(Image.open(path + image).resize((32, 32)))
        
    #need to be normalized before putting into network
    images = np.array(images)
    print(len(images))    
    
    path = LOG_DIR_DATA + "/dead_leaves-mixed_SVHN.npy"
    np.save(path, images)
     """
    path = LOG_DIR_DATA + "/dead_leaves-mixed_SVHN.npy" 
     
    images= np.load(path)
    
    mean = images.mean()
    std = images.std()
    
    train_data = [(torch.FloatTensor((images[i]- mean)/std).permute(2, 0, 1), torch.tensor(0)) for i in range(len(images))]    
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, train_loader, train_loader
    
def get_Shaders21k_SVHN_PATE(batch_size):
    num_workers=4
    """ path = LOG_DIR_DATA + "/shaders21k/"
    
    images=[]
    for image in os.listdir(path):
        images.append(Image.open((path + image)).resize((32, 32)))
        
    #need to be normalized before putting into network
    images = np.array(images) 
    
    
    path = LOG_DIR_DATA + "/Shaders21k_SVHN.npy"
    np.save(path, images)
     """
    path = LOG_DIR_DATA + "/Shaders21k_SVHN.npy"
    images = np.load(path)
    
    mean = images.mean()
    std = images.std()
    
    num_points = 100000
    if len(images) < num_points:
        num_points = len(images)
    
    train_data = [(torch.FloatTensor((images[i]- mean)/std).permute(2, 0, 1), torch.tensor(0)) for i in range(num_points)]
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, train_loader, train_loader
    
def get_stylegan_SVHN_PATE(batch_size):
    num_workers=4
    path = LOG_DIR_DATA + "/stylegan-oriented/"
    
    images=[]
    
    for image in os.listdir(path):
        images.append(Image.open(path + image).resize((32, 32)))
        
    #need to be normalized before putting into network
    images = np.array(images) 
    
    path = LOG_DIR_DATA + "/stylegan_SVHN.npy"
    np.save(path, images)
    
    path = LOG_DIR_DATA + "/stylegan_SVHN.npy" 
     
    images= np.load(path)
    
    mean = images.mean()
    std = images.std()
    
    train_data = [(torch.FloatTensor((images[i]- mean)/std).permute(2, 0, 1), torch.tensor(0)) for i in range(len(images))]
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, train_loader, train_loader
#these datasets are for training the student, they need the teacher_labels saved in the folder /teacher_labels/ to work
#


def get_CIFAR10_student(batch_size, validation_size=0.2):
    num_workers = 4
    
    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49139969, 0.48215842, 0.44653093), (0.24703223,0.24348513, 0.26158784)), #(0.2023, 0.1994, 0.2010)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49421429, 0.4851314, 0.45040911), (0.24665252, 0.24289226, 0.26159238)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.CIFAR10(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test) #, transform=transform_test
    
    end = int(len(testset)*(1-validation_size))
    #end = len(testset)
    
    target_path = LOG_DIR_DATA + "/teacher_labels/CIFAR10.npy"
    
    teacher_labels = np.load(target_path)
    
    partition_train = [[testset[i][0], torch.tensor(teacher_labels[i])] for i in range(end) if teacher_labels[i]!= -1] #remove all datapoints, where we have no answer from the teacher ensemble
    partition_test = [testset[i] for i in range(end, len(testset))]
        
    print("Number of CIFAR10 samples for student training: {}".format(len(partition_train)))
    
    train_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, valid_loader, test_loader

def get_noise_CIFAR10_student(batch_size):
    num_workers = 4

    path = LOG_DIR_DATA + "/noise_CIFAR10.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/noise_CIFAR10.npy"

    dataset = np.load(path)
    targets = np.load(target_path)

    assert len(dataset) == len(targets), "size of dataset and teacher labels does not match"

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49421429, 0.4851314, 0.45040911), (0.24665252, 0.24289226, 0.26159238)),
    ])

    testset = torchvision.datasets.CIFAR10(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)

    trainset = [(torch.FloatTensor(dataset[i]), torch.tensor(targets[i])) for i in range(len(dataset)) if targets[i] != -1] #also need to recheck if we need this

    print("Number of samples for student training: {}".format(len(trainset)))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


    return train_loader, valid_loader, test_loader
     
def get_MNIST_student(batch_size, validation_size=0.2):
    num_workers = 4
    
    transform_train=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

    trainset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    end_valid = int(len(trainset)*(1-validation_size))
    end = int(len(testset)*(1-validation_size))
    
    target_path = LOG_DIR_DATA + "/teacher_labels/MNIST.npy"
    
    teacher_labels = np.load(target_path)
    
    partition_train = [[testset[i][0], torch.tensor(teacher_labels[i])] for i in range(end) if teacher_labels[i]!= -1] #remove all datapoints, where we have no answer from the teacher ensemble
    partition_test = [testset[i] for i in range(end, len(testset))]
    
    partition_valid = [trainset[i] for i in range(10000)]
    
    train_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(partition_valid, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, valid_loader, test_loader

def get_noise_MNIST_student(batch_size, validation_size=0.2):
    num_workers = 4
    
    path = LOG_DIR_DATA + "/noise_MNIST.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    
    dataset = np.load(path)
    targets = np.load(target_path)
    
    assert len(dataset) == len(targets), "size of dataset and teacher labels does not match"
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor(dataset[i]).unsqueeze(0), torch.tensor(targets[i])) for i in range(len(dataset)) if targets[i] != -1] #also need to recheck if we need this
    
    print("Number of samples for student training: {}".format(len(trainset)))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    
    return train_loader, valid_loader, test_loader

def get_FMNIST_student(batch_size, validation_size=0.2):
    num_workers = 4
    
    transform_train=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.2860,), (0.3530,)) # normalize inputs
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.2860,), (0.3530,)) # normalize inputs
    ])

    trainset = torchvision.datasets.FashionMNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.FashionMNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    #end = int(len(trainset)*(1-validation_size))
    end=len(trainset)
    
    target_path = LOG_DIR_DATA + "/teacher_labels/FMNIST.npy"
    
    teacher_labels = np.load(target_path)
    
    partition_train = [[trainset[i][0], torch.tensor(teacher_labels[i])] for i in range(end) if teacher_labels[i]!= -1] #remove all datapoints, where we have no answer from the teacher ensemble
    partition_test = [testset[i] for i in range(len(testset))]
        
    print("Number of samples for student training: {}".format(len(partition_train)))
    
    
    train_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, valid_loader, test_loader

def get_SVHN_student(batch_size, validation_size=0.2):
    num_workers = 4
    

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.44921386, 0.4496643, 0.45029628), (0.20032172, 0.19916263, 0.19936596)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.45207793, 0.45359373, 0.45602703), (0.22993235, 0.229334, 0.2311905)),
    ])
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="train", download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="test", download=True, transform=transform_test) #, transform=transform_test

    end = int(len(testset)*(1-validation_size))
    
    target_path = LOG_DIR_DATA + "/teacher_labels/SVHN.npy"
    
    teacher_labels = np.load(target_path)
    
    partition_train = [[testset[i][0], torch.tensor(teacher_labels[i])] for i in range(end) if teacher_labels[i]!= -1] #remove all datapoints, where we have no answer from the teacher ensemble
    partition_test = [testset[i] for i in range(end, len(testset))]
    
    print("Number of SVHN samples for student training: {}".format(len(partition_train)))
    
    train_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, valid_loader, test_loader

def get_noise_SVHN_student(batch_size, validation_size=0.2):
    num_workers = 4
    
    path = LOG_DIR_DATA + "/noise_SVHN.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/noise_SVHN.npy"
    
    dataset = np.load(path)
    targets = np.load(target_path)
    
    assert len(dataset) == len(targets), "size of dataset and teacher labels does not match"
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.45207793, 0.45359373, 0.45602703), (0.22993235, 0.229334, 0.2311905)),
    ])
    
    testset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="test", download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor(dataset[i]), torch.tensor(targets[i])) for i in range(len(dataset)) if targets[i] != -1] #also need to recheck if we need this
    
    print("Number of noisy samples for student training: {}".format(len(trainset)))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    
    return train_loader, valid_loader, test_loader

def get_dead_leaves_student(batch_size, validation_size=0.2):
    num_workers = 4
    
    path = LOG_DIR_DATA + "/dead_leaves-mixed.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/dead_leaves.npy"
    
    targets = np.load(target_path)
    
    images = np.load(path)
    
    """ #load .jpg dead_leave images and turn into grayscale and reduce dimension so it can be used for MNIST
    for image in os.listdir(path):
        images.append(ImageOps.grayscale(Image.open((path + image))).resize((28, 28)))
        
    #need to be normalized before putting into network
    images = np.array(images) """
    
    assert len(images) == len(targets)
    mean = images.mean()
    std = images.std()
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor((images[i]- mean)/std).unsqueeze(0), torch.tensor(targets[i])) for i in range(len(images)) if targets[i] != -1] #also need to recheck if we need this
    
    print("Number of samples for student training: {}".format(len(trainset)))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    
    return train_loader, valid_loader, test_loader

def get_stylegan_student(batch_size, validation_size=0.2):
    num_workers=4


    path = LOG_DIR_DATA + "/stylegan-oriented.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/stylegan.npy"
    
    targets = np.load(target_path)
    
    images = np.load(path)
    
    """ #load .jpg dead_leave images and turn into grayscale and reduce dimension so it can be used for MNIST
    for image in os.listdir(path):
        images.append(ImageOps.grayscale(Image.open((path + image))).resize((28, 28)))
        
    #need to be normalized before putting into network
    images = np.array(images) """
    
    assert len(images) == len(targets)
    mean = images.mean()
    std = images.std()
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor((images[i]- mean)/std).unsqueeze(0), torch.tensor(targets[i])) for i in range(len(images)) if targets[i] != -1] #also need to recheck if we need this
    
    print("Number of samples for student training: {}".format(len(trainset)))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    
    return train_loader, valid_loader, test_loader

def get_SVHN_MNIST_student(batch_size, validation_size=0.2):
    
    num_workers = 4

    trainset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="train", download=True) #, transform=transform_train

    images = [Image.fromarray(np.transpose(image, (1, 2, 0))) for image in trainset.data]
    
    gray_images = [ImageOps.grayscale(image.resize((28, 28))) for image in images]
    
    gray_images = np.array(gray_images)
    mean = gray_images.mean()
    std = gray_images.std()

    target_path = LOG_DIR_DATA + "/teacher_labels/SVHN_MNIST.npy"
    
    targets = np.load(target_path)
    
    """ #load .jpg dead_leave images and turn into grayscale and reduce dimension so it can be used for MNIST
    for image in os.listdir(path):
        images.append(ImageOps.grayscale(Image.open((path + image))).resize((28, 28)))
        
    #need to be normalized before putting into network
    images = np.array(images) """
    
    assert len(gray_images) == len(targets)
    mean = gray_images.mean()
    std = gray_images.std()
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor((gray_images[i]- mean)/std).unsqueeze(0), torch.tensor(targets[i])) for i in range(len(gray_images)) if targets[i] != -1] #also need to recheck if we need this
    
    print("Number of samples for student training: {}".format(len(trainset)))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    
    return train_loader, valid_loader, test_loader

def get_FractalDB_student(batch_size, validation_size=0.2):
    num_workers=4


    path = LOG_DIR_DATA + "/FractalDB.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/FractalDB.npy"
    
    targets = np.load(target_path)
    
    images = np.load(path)
    
    """ #load .jpg dead_leave images and turn into grayscale and reduce dimension so it can be used for MNIST
    for image in os.listdir(path):
        images.append(ImageOps.grayscale(Image.open((path + image))).resize((28, 28)))
        
    #need to be normalized before putting into network
    images = np.array(images) """
    
    assert len(images) == len(targets)
    mean = images.mean()
    std = images.std()
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor((images[i]- mean)/std).unsqueeze(0), torch.tensor(targets[i])) for i in range(len(images)) if targets[i] != -1] #also need to recheck if we need this
    
    print("Number of samples for student training: {}".format(len(trainset)))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    
    return train_loader, valid_loader, test_loader

def get_Shaders21k_student(batch_size, validation_size=0.2):
    num_workers=4


    path = LOG_DIR_DATA + "/Shaders21k.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/Shaders21k.npy"
    
    targets = np.load(target_path)
    
    images = np.load(path)
    
    """ #load .jpg dead_leave images and turn into grayscale and reduce dimension so it can be used for MNIST
    for image in os.listdir(path):
        images.append(ImageOps.grayscale(Image.open((path + image))).resize((28, 28)))
        
    #need to be normalized before putting into network
    images = np.array(images) """
    
    assert 100000 == len(targets)
    mean = images.mean()
    std = images.std()
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    num_points = 100000
    if len(images) < num_points:
        num_points = len(images)
        
    trainset = [(torch.FloatTensor((images[i]- mean)/std).unsqueeze(0), torch.tensor(targets[i])) for i in range(num_points) if targets[i] != -1] #also need to recheck if we need this
    
    print("Number of samples for student training: {}".format(len(trainset)))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    
    return train_loader, valid_loader, test_loader

def get_MIX_student(batch_size, validation_size=0.2):
    num_workers = 4
    path = LOG_DIR_DATA + "/MIX.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/MIX.npy"
    
    targets = np.load(target_path)
    
    images = np.load(path)
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor(images[i]), torch.tensor(targets[i])) for i in range(len(images)) if targets[i] != -1]
    
    print("Number of samples for student training: {}".format(len(trainset)))
    
    datasets = ["noise", "FMNIST", "leaves", "StyleGAN", "FractalDB", "Shaders"]
    
    
    for i, dataset in enumerate(datasets):
        current =targets[5000*i:5000*(i+1)]
        num_answered = (current != -1).sum()
        print(f"{dataset} had {num_answered} queries answered.")
    
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    
    return train_loader, valid_loader, test_loader
    
def get_dead_leaves_SVHN_student(batch_size, validation_size=0.2):
    
    num_workers = 4
    path = LOG_DIR_DATA + "/dead_leaves-mixed_SVHN.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/dead_leaves_SVHN.npy"
    
    targets = np.load(target_path)
    
    images = np.load(path)
    
    num_points = 100000
    if len(images) < num_points:
        num_points = len(images)

    mean = images.mean()
    std = images.std()
    trainset = [(torch.FloatTensor((images[i]- mean)/std).permute(2, 0, 1), torch.tensor(targets[i])) for i in range(num_points) if targets[i] != -1]
    
    print("Number of samples for student training: {}".format(len(trainset)))
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.45207793, 0.45359373, 0.45602703), (0.22993235, 0.229334, 0.2311905)),
    ])
    
    testset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="test", download=True, transform=transform_test) #, transform=transform_test

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, valid_loader, test_loader

def get_Shaders21k_SVHN_student(batch_size, validation_size=0.2):
    num_workers = 4
    path = LOG_DIR_DATA + "/Shaders21k_SVHN.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/Shaders21k_SVHN.npy"
    
    targets = np.load(target_path)
    
    images = np.load(path)

    mean = images.mean()
    std = images.std()
    trainset = [(torch.FloatTensor((images[i]- mean)/std).permute(2, 0, 1), torch.tensor(targets[i])) for i in range(len(images)) if targets[i] != -1]
    
    print("Number of samples for student training: {}".format(len(trainset)))
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.45207793, 0.45359373, 0.45602703), (0.22993235, 0.229334, 0.2311905)),
    ])
    
    testset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="test", download=True, transform=transform_test) #, transform=transform_test

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, valid_loader, test_loader
    
def get_stylegan_SVHN_student(batch_size, validation_size=0.2):
    
    num_workers = 4
    path = LOG_DIR_DATA + "/stylegan_SVHN.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/stylegan_SVHN.npy"
    
    targets = np.load(target_path)
    
    images = np.load(path)

    mean = images.mean()
    std = images.std()
    trainset = [(torch.FloatTensor((images[i]- mean)/std).permute(2, 0, 1), torch.tensor(targets[i])) for i in range(len(images)) if targets[i] != -1]
    
    print("Number of samples for student training: {}".format(len(trainset)))
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.45207793, 0.45359373, 0.45602703), (0.22993235, 0.229334, 0.2311905)),
    ])
    
    testset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="test", download=True, transform=transform_test) #, transform=transform_test

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, valid_loader, test_loader



""" for image in os.listdir(path):
        images.append(ImageOps.grayscale(Image.open((path + image))).resize((28, 28)))
        
    #need to be normalized before putting into network
    images = np.array(images) """