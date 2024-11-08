from torchmetrics.image.fid import FrechetInceptionDistance
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