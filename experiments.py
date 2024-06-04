import pate_data
from utils import help, misc
import datasets, conventions
import student
import models

import torchvision
import torchvision.transforms as transforms
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR_DATA = "/disk2/michel/data"
LOG_DIR = "/disk2/michel/"

def compare_ensemble_output(length):
    
    noise_vote_array_path = LOG_DIR_DATA +  "/vote_array/noise_MNIST.npy"

    noise_label_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"

    noise_vote_array = np.load(noise_vote_array_path)
    vote_array=noise_vote_array.T
    
    t,l,r = datasets.get_noise_MNIST_PATE(256)
    
    true_labels=[]
    
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("noise_MNIST")
    teacher_name = conventions.resolve_teacher_name(experiment_config)
    teacher_path = os.path.join("/disk2/michel", "Pretrained_NW","MNIST", teacher_name)
    teacher_nw = torch.load(teacher_path)
    teacher_nw.to(device)
    teacher_nw.train()
    
        
    for data, _ in t:
        data = data.to(device)
        with torch.no_grad():
            teacher_output = teacher_nw(data)   
        label = np.argmax(teacher_output.cpu().numpy(), axis=1)
        for j in label:
            true_labels.append(j)
      
    num_samples = vote_array.shape[0]
          
    all_labels = []
    
    for i in range(len(vote_array)):
        uniques, counts = np.unique(vote_array[i], return_counts=True)
        all_labels.append(dict(zip(uniques, counts)))
    
    
    predicted_labels = np.zeros(num_samples)
    
    for i in range(len(vote_array)):
        predicted_labels[i] = np.bincount(vote_array[i]).argmax()
    
    for i in range(length):
        print("Reference label: {}, Ensemble predictions: ".format(true_labels[i]), end="")
        output = ", ".join(f'{number}: {count}' for number, count in all_labels[i].items())
        print(output, end = "")
        print(" Predicted label by ensemble: {}".format(int(predicted_labels[i])))
        
        
def plot_count_histogram(title="consensus_same_init.png", votearray_path="/disk2/michel/data/vote_array/noise_MNIST.npy"):
    """Experiment, to plot the histograms of the ensemble consensus. The idea is to then train teachers with the same initialization and check if the consensus changes.

    Args:
        title (str, optional): Title of the histogram.
        votearray_path (str, optional): Path to the vote array, which is output by the teachers. Defaults to "/disk2/michel/data/vote_array/noise_MNIST.npy".
    """
    
    noise_vote_array = np.load(votearray_path)
    vote_array=noise_vote_array.T
          
    histogram_values = []
    
    for i in range(len(vote_array)):
        uniques, counts = np.unique(vote_array[i], return_counts=True)
        count_most_frequent = counts[np.argmax(counts)]
        
        histogram_values.append(count_most_frequent)
        
    plt.hist(histogram_values, bins=240, density=True)
    plt.ylabel("Occurence of highest counts")
    plt.xlabel("Highest consensus of teachers")
    plt.title("Consensus of teachers")
    plt.savefig(os.path.join(LOG_DIR, 'Plots', title), dpi=200)
    
    
    
def use_histogram():
    
    batch_size=256
    num_workers=4
    
    
    noise_vote_array = np.load("/disk2/michel/data/vote_array/noise_MNIST.npy")
    noise_vote_array = noise_vote_array.T
    
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    pate_labels = np.load(noise_label_path)
    
    targets = pate_data.create_histogram_labels(noise_vote_array)
    
    path = LOG_DIR_DATA + "/noise_MNIST.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    
    dataset = np.load(path)
    
    assert len(dataset) == len(targets), "size of dataset and teacher labels does not match"
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor(dataset[i]).unsqueeze(0), torch.tensor(targets[i])) for i in range(12000)] #use all available datapoints which have been answered by teachers
    
    print(len(trainset))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("noise_MNIST")
    # override
    
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs('/disk2/michel/data', exist_ok=True)
    os.makedirs('/disk2/michel/Pretrained_NW/{}'.format("noise_MNIST"), exist_ok=True)
    
    student_model = model = eval("models.{}.Target_Net({}, {})".format(
        experiment_config["model_student"],
        experiment_config["inputs"],
        experiment_config["code_dim"]
    )).to(device)
    
    metrics = student.train_student(student_model, train_loader, valid_loader, 100, 0.001, 0, True, device, False, LOG_DIR, optim="Adam", label=False)
       
    model_name = conventions.resolve_student_name(experiment_config)
    torch.save(model, os.path.join('/disk2/michel/Pretrained_NW/{}'.format("noise_MNIST"), model_name))
    
    plt.plot(range(1, len(metrics[1])+1), metrics[1], label="Train Accuracy")
    plt.plot(range(1, len(metrics[3])+1), metrics[3], label="Valid Accuracy")
    plt.title('Student Training')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy_student.png'), dpi=200)
    plt.close()

    plt.plot(range(1, len(metrics[0])+1), metrics[0], label="Train Loss")
    plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Valid Loss")

    plt.title('Student Training')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'loss_student.png'), dpi=200)
    plt.close()
    print("Student training is finished.")
    
def use_logits():
    
    batch_size=256
    num_workers=4
    
    
    
    #noise_vote_array = pate_data.query_teachers_softmax("noise_MNIST", 200)
    
    noise_vote_array = np.load("/disk2/michel/data/logit_array/noise_MNIST.npy")
    noise_vote_array = noise_vote_array.T
    
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    pate_labels = np.load(noise_label_path)
    
    targets = pate_data.create_logit_labels(noise_vote_array)
    
    path = LOG_DIR_DATA + "/noise_MNIST.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    
    dataset = np.load(path)
    
    assert len(dataset) == len(targets), "size of dataset and teacher labels does not match"
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor(dataset[i]).unsqueeze(0), torch.tensor(targets[i])) for i in range(12000)] #use all available datapoints which have been answered by teachers
    
    print(len(trainset))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("noise_MNIST")
    # override
    
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs('/disk2/michel/data', exist_ok=True)
    os.makedirs('/disk2/michel/Pretrained_NW/{}'.format("noise_MNIST"), exist_ok=True)
    
    student_model = model = eval("models.{}.Target_Net({}, {})".format(
        experiment_config["model_student"],
        experiment_config["inputs"],
        experiment_config["code_dim"]
    )).to(device)
    
    metrics = student.train_student(student_model, train_loader, valid_loader, 100, 0.001, 0, True, device, False, LOG_DIR, optim="Adam", label=False)
       
    model_name = conventions.resolve_student_name(experiment_config)
    torch.save(model, os.path.join('/disk2/michel/Pretrained_NW/{}'.format("noise_MNIST"), model_name))
    
    plt.plot(range(1, len(metrics[1])+1), metrics[1], label="Train Accuracy")
    plt.plot(range(1, len(metrics[3])+1), metrics[3], label="Valid Accuracy")
    plt.title('Student Training')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy_student.png'), dpi=200)
    plt.close()

    plt.plot(range(1, len(metrics[0])+1), metrics[0], label="Train Loss")
    plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Valid Loss")

    plt.title('Student Training')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'loss_student.png'), dpi=200)
    plt.close()
    print("Student training is finished.")
    
    
use_logits()