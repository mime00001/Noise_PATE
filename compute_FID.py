from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import torchvision.transforms as transforms
import torchvision
import numpy as np
import torch

LOG_DIR_DATA = "/storage3/michel/data"
LOG_DIR = "/storage3/michel"




def calculate_FID(dataset1_base, dataset2_dist):

    fid = FrechetInceptionDistance(feature=64).to("cuda")
    
    fid.update(dataset1_base, real=True)
    fid.update(dataset2_dist, real=False)
    return fid.compute()
    
def calculate_KID(dataset1_base, dataset2_dist):

    kid = KernelInceptionDistance(subsets=10, subset_size=500).to("cuda")
    
    kid.update(dataset1_base, real=True)
    kid.update(dataset2_dist, real=False)
    return kid.compute()
    


def prep_MNIST_test(length=500):
    transform_test = transforms.Compose([
            transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    testset = testset.data.unsqueeze(1)
    testset = testset.repeat(1,3, 1, 1)
    
    rgb_testset = []
    for i in range(length):
        image=testset[i]
        
        rgb_array = torch.tensor(image, dtype=torch.uint8)
        rgb_testset.append(rgb_array)
    rgb_testset = torch.stack(rgb_testset)
    return rgb_testset

def prep_MNIST_train(length=500):
    transform_test = transforms.Compose([
            transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_test)
    
    testset = testset.data.unsqueeze(1)
    testset = testset.repeat(1,3, 1, 1)
    
    rgb_testset = []
    for i in range(length):
        image=testset[i]
        
        rgb_array = torch.tensor(image, dtype=torch.uint8)
        rgb_testset.append(rgb_array)
    rgb_testset = torch.stack(rgb_testset)
    return rgb_testset



def prep_SVHN_test(length=500):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.45207793, 0.45359373, 0.45602703), (0.22993235, 0.229334, 0.2311905)),
    ])
    testset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="test", download=True, transform=transform_test)
    
    testset = testset.data
    
    rgb_testset = []
    for i in range(length):
        image=testset[i]
        
        rgb_array = torch.tensor(image, dtype=torch.uint8) #.permute(2, 0, 1)
        rgb_testset.append(rgb_array)
    rgb_testset = torch.stack(rgb_testset)
    return rgb_testset

def prep_SVHN_train(length=500):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.44921386, 0.4496643, 0.45029628), (0.20032172, 0.19916263, 0.19936596)),
    ])
    
    trainset = torchvision.datasets.SVHN(root=LOG_DIR_DATA, split="train", download=True, transform=transform_train)
    
    trainset = trainset.data
    
    rgb_testset = []
    for i in range(length):
        image=trainset[i]
        
        rgb_array = torch.tensor(image, dtype=torch.uint8) #.permute(2, 0, 1)
        rgb_testset.append(rgb_array)
    rgb_testset = torch.stack(rgb_testset)
    return rgb_testset

def prep_dataset(datasetname, length=500):
    path = LOG_DIR_DATA + "/{}.npy".format(datasetname)
    
    data = np.load(path)
    mean = data.mean()
    std = data.std()
    
    traindata =  [torch.tensor((data[i]- mean)/std, dtype=torch.uint8).unsqueeze(0) for i in range(length)]
    traindata = torch.stack(traindata)
    print(traindata.shape)
    #traindata = traindata.unsqueeze(1)
    traindata = traindata.repeat(1, 3, 1, 1)
    

    return traindata

def prep_RGB_dataset(datasetname, length=500):
    path = LOG_DIR_DATA + "/{}.npy".format(datasetname)
    
    data = np.load(path)
    mean = data.mean()
    std = data.std()
    if datasetname != "noise_SVHN":
        traindata = [torch.tensor(((data[i]-mean) / std), dtype=torch.uint8).permute(2, 0, 1) for i in range(length)] #
    else: 
        traindata = [torch.tensor(((data[i]-mean) / std), dtype=torch.uint8) for i in range(length)] #
    
    traindata = torch.stack(traindata)
    return traindata

def prep_FMNIST_test(length=500):
    transform_test = transforms.Compose([
            transforms.ToTensor(), # first, convert image to PyTorch tensor
            transforms.Normalize((0.2860,), (0.3530,)) # normalize inputs
    ])
    testset = torchvision.datasets.FashionMNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    rgb_testset = []
    testset = testset.data.unsqueeze(1)
    testset = testset.repeat(1, 3, 1, 1)
    for i in range(length):
        image=testset[i]
        
        rgb_array = torch.tensor(image, dtype=torch.uint8)
        rgb_testset.append(rgb_array)
    rgb_testset = torch.stack(rgb_testset)
    return rgb_testset

def prep_FMNIST_train(length=500):
    transform_train=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.2860,), (0.3530,)) # normalize inputs
    ])

    trainset = torchvision.datasets.FashionMNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train)
    rgb_testset = []
    trainset = trainset.data.unsqueeze(1)
    trainset = trainset.repeat(1, 3, 1, 1)
    for i in range(length):
        image=trainset[i]
        
        rgb_array = torch.tensor(image, dtype=torch.uint8)
        rgb_testset.append(rgb_array)
    rgb_testset = torch.stack(rgb_testset)
    return rgb_testset

def FID_MNIST():
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

    trainset = trainset.data.unsqueeze(1)
    testset = testset.data.unsqueeze(1)
    trainset = trainset.repeat(1, 3, 1, 1)
    testset = testset.repeat(1, 3, 1, 1)
    
    rgb_trainset = []
    rgb_testset = []
    
    length = 500
    
    for i in range(length):
        image = trainset[i]
       
        rgb_array = torch.tensor(image, dtype=torch.uint8)
        rgb_trainset.append(rgb_array)
    rgb_trainset = torch.stack(rgb_trainset)
    for i in range(length):
        image=testset[i]
        
        rgb_array = torch.tensor(image, dtype=torch.uint8)
        rgb_testset.append(rgb_array)
    rgb_testset = torch.stack(rgb_testset)

    print(calculate_FID(rgb_trainset.to("cuda"), rgb_testset.to("cuda")))
