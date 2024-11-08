from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
import torchvision
import numpy as np
import torch

LOG_DIR_DATA = "/storage3/michel/data"
LOG_DIR = "/storage3/michel"




def calculate_FID(dataset1_base, dataset2_dist):

    fid = FrechetInceptionDistance(feature=2048)
    
    fid.update(dataset1_base, real=True)
    fid.update(dataset2_dist, real=False)
    return fid.compute()
    
    

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

trainset = trainset.data
testset = testset.data
rgb_trainset = []
rgb_testset = []
for image in trainset:
    
    rgb_array = np.stack([image]*3, axis=-1)
    rgb_trainset.append(rgb_array)
rgb_trainset = torch.stack(rgb_trainset)
for image in testset:
    
    rgb_array = np.stack([image]*3, axis=-1)
    rgb_testset.append(rgb_array)
rgb_testset = torch.stack(rgb_testset)



print(calculate_FID(rgb_trainset, rgb_testset))
