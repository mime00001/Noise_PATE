import models.resnet10
import models.resnet12
import models.resnet9
import models.mnistresnet
import datasets
import torchvision
import torchvision.transforms as transforms
from utils import teachers
from pate_data import query_teachers
import models
import torch
import numpy as np

LOG_DIR_DATA = "/disk2/michel/data"

path = LOG_DIR_DATA + "/teacher_labels/MNIST.npy"

a = np.load(path)
print(a.shape)
#query_teachers("MNIST", 200)


#kd lr=0.001, decay =0
#CAPC epochs=500, nb_teachers = 50, lr=0.01, decay = 1e-5

#for mnist epochs = 35, lr = 0.001, decay = 0, sigma1 = 150, sigma2 = 50, delta = 10e-8, epsilon =3, T=200