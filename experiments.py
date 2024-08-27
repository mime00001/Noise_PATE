import torch.utils
import torch.utils.data
import pate_main
import pate_data
from utils import help, misc, teachers
import datasets, conventions
import student
import models
import distill_gaussian

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

#this file contains experiments and helper functions

LOG_DIR_DATA = "/storage3/michel/data"
LOG_DIR = "/storage3/michel"

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
    teacher_path = os.path.join(LOG_DIR, "Pretrained_NW","MNIST", teacher_name)
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
        
        
def plot_count_histogram(title="consensus_same_init_SVHN.png", votearray_path="/storage3/michel/data/vote_array/noise_MNIST.npy", ylim=0.05, histogram_values_path=None):
    """Experiment, to plot the histograms of the ensemble consensus. The idea is to then train teachers with the same initialization and check if the consensus changes.

    Args:
        title (str, optional): Title of the histogram.
        votearray_path (str, optional): Path to the vote array, which is output by the teachers. Defaults to "/disk2/michel/data/vote_array/noise_MNIST.npy".
    """
    if votearray_path:
        noise_vote_array = np.load(votearray_path)
        vote_array=noise_vote_array.T
            
        histogram_values = []
        
        for i in range(len(vote_array)):
            uniques, counts = np.unique(vote_array[i], return_counts=True)
            count_most_frequent = counts[np.argmax(counts)]
            
            histogram_values.append(count_most_frequent)
    else:
        histogram_values = np.load(histogram_values_path)
    
        
    plt.hist(histogram_values, bins=240, density=True)
    plt.ylim(0, 0.3)
    plt.xlim(20, 200) 
    plt.ylabel("Occurence")
    plt.xlabel("Number of teachers that agree on final label")
    plt.title("Consensus of teachers")
    plt.savefig(os.path.join(LOG_DIR, 'Plots', title), dpi=200)
    
    arr = np.array(histogram_values)
    save_path = title.replace(".png", "")
    np.save(os.path.join(LOG_DIR, 'Plots', save_path), arr)
    
    
def use_histogram():
    
    batch_size=256
    num_workers=4
    
    
    noise_vote_array = np.load("/storage3/michel/data/vote_array/noise_MNIST.npy")
    noise_vote_array = noise_vote_array.T
    
    targets = pate_data.create_histogram_labels(noise_vote_array)
    
    path = LOG_DIR_DATA + "/noise_MNIST.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    
    teacher_labels = np.load(target_path)
    
    dataset = np.load(path)
    
    assert len(dataset) == len(targets), "size of dataset and teacher labels does not match"
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor(dataset[i]).unsqueeze(0), torch.tensor(targets[i])) for i in range(len(dataset)) if teacher_labels[i] != -1] #use all available datapoints which have been answered by teachers
    
    num_samples = len(trainset)
    print(len(trainset))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("noise_MNIST")
    # override
    
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs('/storage3/michel/data', exist_ok=True)
    os.makedirs('/storage3/michel/Pretrained_NW/{}'.format("noise_MNIST"), exist_ok=True)
    
    student_model = model = eval("models.{}.Target_Net({}, {})".format(
        experiment_config["model_student"],
        experiment_config["inputs"],
        experiment_config["code_dim"]
    )).to(device)
    
    metrics = student.train_student(student_model, train_loader, valid_loader, 50, 0.0001, 0, True, device, False, LOG_DIR, optim="Adam", label=False, loss="misc")
    
    plt.ylim(0, 1) 
    plt.plot(range(1, len(metrics[1])+1), metrics[1], label="Train Accuracy")
    plt.plot(range(1, len(metrics[3])+1), metrics[3], label="Valid Accuracy")
    plt.title('Student Training using histogram and {} samples'.format(num_samples))
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy_student_histogram.png'), dpi=200)
    plt.close()

    plt.plot(range(1, len(metrics[0])+1), metrics[0], label="Train Loss")
    plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Valid Loss")

    plt.title('Student Training using histogram and {} samples'.format(num_samples))
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'loss_student_histogram.png'), dpi=200)
    plt.close()
    print("Student training using histogram is finished.")
    
    return metrics[3][-1]
    
def use_logits():
    
    batch_size=256
    num_workers=4
    
    
    
    #noise_vote_array = pate_data.query_teachers_logits("noise_MNIST", 200)
    
    noise_vote_array = np.load("/storage3/michel/data/logit_array/noise_MNIST.npy")
    noise_vote_array = np.transpose(noise_vote_array, (1, 0, 2))
    
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    pate_labels = np.load(noise_label_path)
    
    targets = pate_data.create_logit_labels(noise_vote_array)
    
    path = LOG_DIR_DATA + "/noise_MNIST.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    
    
    teacher_labels = np.load(target_path)
    dataset = np.load(path)
    
    assert len(dataset) == len(targets), "size of dataset and teacher labels does not match"
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor(dataset[i]).unsqueeze(0), torch.tensor(targets[i])) for i in range(len(dataset)) if teacher_labels[i] != -1] #use all available datapoints which have been answered by teachers
    
    #trainset = [(torch.FloatTensor(dataset[i]).unsqueeze(0), torch.tensor(targets[i])) for i in range(12000)]
    
    num_samples = len(trainset)
    
    print(len(trainset))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("noise_MNIST")
    # override
    
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs(LOG_DIR_DATA, exist_ok=True)
    os.makedirs(LOG_DIR +'/Pretrained_NW/{}'.format("noise_MNIST"), exist_ok=True)
    
    student_model = model = eval("models.{}.Target_Net({}, {})".format(
        experiment_config["model_student"],
        experiment_config["inputs"],
        experiment_config["code_dim"]
    )).to(device)
    
    metrics = student.train_student(student_model, train_loader, valid_loader, 50, 0.001, 0, True, device, False, LOG_DIR, optim="Adam", label=False, loss="misc")
       
    plt.ylim(0, 1)
    plt.plot(range(1, len(metrics[1])+1), metrics[1], label="Train Accuracy")
    plt.plot(range(1, len(metrics[3])+1), metrics[3], label="Valid Accuracy")
    
    plt.title('Student Training using logits and {} samples'.format(num_samples))
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy_student_logits.png'), dpi=200)
    plt.close()

    plt.plot(range(1, len(metrics[0])+1), metrics[0], label="Train Loss")
    plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Valid Loss")

    plt.title('Student Training using logits and {} samples'.format(num_samples))
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'loss_student_logits.png'), dpi=200)
    plt.close()
    print("Student training using logits is finished.")
    
    return metrics[3][-1]

def use_softmax():
    
    batch_size=256
    num_workers=4
    
    
    
    #noise_vote_array = pate_data.query_teachers_logits("noise_MNIST", 200)
    
    noise_vote_array = np.load(LOG_DIR_DATA + "/logit_array/noise_MNIST.npy")
    noise_vote_array = np.transpose(noise_vote_array, (1, 0, 2))
    
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    pate_labels = np.load(noise_label_path)
    
    targets = pate_data.create_softmax_labels(noise_vote_array)
    
    path = LOG_DIR_DATA + "/noise_MNIST.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    
    
    teacher_labels = np.load(target_path)
    dataset = np.load(path)
    
    assert len(dataset) == len(targets), "size of dataset and teacher labels does not match"
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor(dataset[i]).unsqueeze(0), torch.tensor(targets[i])) for i in range(len(dataset)) if teacher_labels[i] != -1] #use all available datapoints which have been answered by teachers
    
    #trainset = [(torch.FloatTensor(dataset[i]).unsqueeze(0), torch.tensor(targets[i])) for i in range(12000)]
    
    num_samples = len(trainset)
    
    print(len(trainset))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("noise_MNIST")
    # override
    
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs(LOG_DIR_DATA, exist_ok=True)
    os.makedirs(LOG_DIR + '/Pretrained_NW/{}'.format("noise_MNIST"), exist_ok=True)
    
    student_model = model = eval("models.{}.Target_Net({}, {})".format(
        experiment_config["model_student"],
        experiment_config["inputs"],
        experiment_config["code_dim"]
    )).to(device)
    
    metrics = student.train_student(student_model, train_loader, valid_loader, 50, 0.001, 0, True, device, False, LOG_DIR, optim="Adam", label=False, loss="softmax")
       
    plt.ylim(0, 1)
    plt.plot(range(1, len(metrics[1])+1), metrics[1], label="Train Accuracy")
    plt.plot(range(1, len(metrics[3])+1), metrics[3], label="Valid Accuracy")
    
    plt.title('Student Training using softmax and {} samples'.format(num_samples))
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy_student_softmax.png'), dpi=200)
    plt.close()

    plt.plot(range(1, len(metrics[0])+1), metrics[0], label="Train Loss")
    plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Valid Loss")

    plt.title('Student Training using softmax and {} samples'.format(num_samples))
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'loss_student_softmax.png'), dpi=200)
    plt.close()
    print("Student training using softmax is finished.")
    
    return metrics[3][-1]


def use_noisy_softmax_label(sigma_gnmax):
    
    batch_size=256
    num_workers=4
    
    
    
    #noise_vote_array = pate_data.query_teachers_logits("noise_MNIST", 200)
    
    noise_vote_array = np.load(LOG_DIR_DATA + "/logit_array/noise_MNIST.npy")
    noise_vote_array = np.transpose(noise_vote_array, (1, 0, 2))
    
    targets = pate_data.get_noisy_softmax_label(noise_vote_array, sigma_gnmax)
    
    path = LOG_DIR_DATA + "/noise_MNIST.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    
    
    teacher_labels = np.load(target_path)
    dataset = np.load(path)
    
    assert len(dataset) == len(targets), "size of dataset and teacher labels does not match"
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor(dataset[i]).unsqueeze(0), torch.tensor(targets[i])) for i in range(len(dataset)) if teacher_labels[i] != -1] #use all available datapoints which have been answered by teachers
    
    #trainset = [(torch.FloatTensor(dataset[i]).unsqueeze(0), torch.tensor(targets[i])) for i in range(12000)]
    
    num_samples = len(trainset)
    
    print(len(trainset))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("noise_MNIST")
    # override
    
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs(LOG_DIR_DATA, exist_ok=True)
    os.makedirs(LOG_DIR + '/Pretrained_NW/{}'.format("noise_MNIST"), exist_ok=True)
    
    student_model = model = eval("models.{}.Target_Net({}, {})".format(
        experiment_config["model_student"],
        experiment_config["inputs"],
        experiment_config["code_dim"]
    )).to(device)
    
    metrics = student.train_student(student_model, train_loader, valid_loader, 50, 0.001, 0, True, device, False, LOG_DIR, optim="Adam", label=True)
       
    plt.ylim(0, 1)
    plt.plot(range(1, len(metrics[1])+1), metrics[1], label="Train Accuracy")
    plt.plot(range(1, len(metrics[3])+1), metrics[3], label="Valid Accuracy")
    
    plt.title('Student Training using softmax and {} samples'.format(num_samples))
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy_student_noisy_softmax.png'), dpi=200)
    plt.close()

    plt.plot(range(1, len(metrics[0])+1), metrics[0], label="Train Loss")
    plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Valid Loss")

    plt.title('Student Training using softmax and {} samples'.format(num_samples))
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'loss_student_noisy_softmax.png'), dpi=200)
    plt.close()
    print("Student training using summed softmax is finished.")
    
    return metrics[3][-1]

def use_ensemble_argmax():
    
    batch_size=256
    num_workers=4
    
    noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/noise_MNIST.npy")
    noise_vote_array = noise_vote_array.T
    
    path = LOG_DIR_DATA + "/noise_MNIST.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    
    teacher_labels = np.load(target_path)
    
    targets = pate_data.get_argmax_labels(noise_vote_array)
    
    dataset = np.load(path)
    
    assert len(dataset) == len(targets), "size of dataset and teacher labels does not match"
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    #trainset = [(torch.FloatTensor(dataset[i]).unsqueeze(0), torch.tensor(targets[i])) for i in range(len(dataset)) if teacher_labels[i] != -1] #use all available datapoints which have been answered by teachers
    
    trainset = [(torch.FloatTensor(dataset[i]).unsqueeze(0), torch.tensor(targets[i])) for i in range(len(dataset)) if teacher_labels[i] != -1]
    
    num_samples = len(trainset)
    print(num_samples)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("noise_MNIST")
    # override
    
    print('Experiment Configuration:')
    print(experiment_config)

    os.makedirs(LOG_DIR_DATA, exist_ok=True)
    os.makedirs(LOG_DIR + '/Pretrained_NW/{}'.format("noise_MNIST"), exist_ok=True)
    
    student_model = model = eval("models.{}.Target_Net({}, {})".format(
        experiment_config["model_student"],
        experiment_config["inputs"],
        experiment_config["code_dim"]
    )).to(device)
    
    metrics = student.train_student(student_model, train_loader, valid_loader, 60, 0.001, 0, True, device, False, LOG_DIR, optim="Adam", label=True)
       
    plt.ylim(0, 1)
    plt.plot(range(1, len(metrics[1])+1), metrics[1], label="Train Accuracy")
    plt.plot(range(1, len(metrics[3])+1), metrics[3], label="Valid Accuracy")
    plt.title('Student Training using argmax over ensemble and {} samples'.format(num_samples))
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'accuracy_student_argmax.png'), dpi=200)
    plt.close()

    plt.plot(range(1, len(metrics[0])+1), metrics[0], label="Train Loss")
    plt.plot(range(1, len(metrics[2])+1), metrics[2], label="Valid Loss")

    plt.title('Student Training using argmax over ensemble and {} samples'.format(num_samples))
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'Plots', 'loss_student_argmax.png'), dpi=200)
    plt.close()
    print("Student training using argmax is finished.")
    
    return metrics[3][-1]
    

def recompute_baseline():
    batch_size =256
    num_workers=4
    
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("noise_MNIST")
    teacher_name = conventions.resolve_teacher_name(experiment_config)
    teacher_path = os.path.join(LOG_DIR, "Pretrained_NW","MNIST", teacher_name)
    teacher_nw = torch.load(teacher_path)
    teacher_nw.to(device)
    
    student_nw = eval("models.{}.Target_Net({}, {})".format(
        experiment_config['model_student'],
        experiment_config['inputs'],
        experiment_config['code_dim']
    )).to(device)
    
    path = LOG_DIR_DATA + "/noise_MNIST.npy"
    target_path = LOG_DIR_DATA + "/teacher_labels/noise_MNIST.npy"
    
    teacher_labels = np.load(target_path)
    
    dataset = np.load(path)
    
    transform_test = transforms.Compose([
         transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    
    testset = torchvision.datasets.MNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    trainset = [(torch.FloatTensor(dataset[i]).unsqueeze(0), torch.tensor(teacher_labels[i])) for i in range(len(dataset)) if teacher_labels[i] != -1]
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    
    distill_gaussian.distill_using_data(None, teacher_nw=teacher_nw, student_nw=student_nw, valid_loader=valid_loader, train_loader=train_loader)
    

def betterFMNIST():
    batch_size = 256
    num_workers = 4
    validation_size = 0.2
    """ transform_train = transform=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.2860,), (0.3530,)) # normalize inputs
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.2860,), (0.3530,)) # normalize inputs
    ])

    trainset = torchvision.datasets.FashionMNIST(root=LOG_DIR_DATA, train=True, download=True, transform=transform_train) #, transform=transform_train
    testset = torchvision.datasets.FashionMNIST(root=LOG_DIR_DATA, train=False, download=True, transform=transform_test)
    
    end = int(len(testset)*(1-validation_size))
    
    partition_train = [testset[i] for i in range(end)]
    partition_test = [testset[i] for i in range(end, len(testset))]
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("MNIST")
    labels = [[] for i in range(200)]
    
    testdata = next(iter(train_loader))[0].numpy()
    for i in range(200):
        print("querying teacher {}".format(i))
        teacher_name = conventions.resolve_teacher_name(experiment_config)
        teacher_name += "_{}".format(i)
        LOG_DIR = '/disk2/michel/Pretrained_NW'
        teacher_path = os.path.join(LOG_DIR, "MNIST", teacher_name)
        teacher_nw = torch.load(teacher_path)
        teacher_nw = teacher_nw.to(device)
        
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
                labels[i].append(j)
     """
    path = LOG_DIR_DATA + "/vote_array/{}".format("FMNIST.npy")
    f_vote_array = np.load(path)

    f_vote_array = f_vote_array.T
    
    path2 = LOG_DIR_DATA + "/teacher_labels/{}".format("FMNIST.npy")

    FMNIST_list=[]
    epsilon_list = [20]
    for eps in epsilon_list:
        achieved_eps, pate_labels = pate_main.inference_pate(vote_array=f_vote_array, threshold=150, sigma_threshold=120, sigma_gnmax=40, epsilon=eps, delta=1e-5, num_classes=10, save=True, savepath=path2)
        final_acc = student.util_train_student(target_dataset="MNIST", transfer_dataset="FMNIST", n_epochs=50, use_test_loader=True, optimizer="Adam")
        FMNIST_list.append((achieved_eps, final_acc))

def normalize_SVHN(trainset, testset):
    data_r =  np.dstack([trainset[i][0][:,:,0] for i in range(len(trainset))])
    data_g =  np.dstack([trainset[i][0][:,:,1] for i in range(len(trainset))])
    data_b =  np.dstack([trainset[i][0][:,:,2] for i in range(len(trainset))])

    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    print(mean)
    print(std)
    
    data_r =  np.dstack([testset[i][0][:,:,0] for i in range(len(testset))])
    data_g =  np.dstack([testset[i][0][:,:,1] for i in range(len(testset))])
    data_b =  np.dstack([testset[i][0][:,:,2] for i in range(len(testset))])

    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    print(mean)
    print(std)
    
def use_pretrained_model():
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    num_classes=10
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    
    print(model)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    print("Network after modifying conv1:")
    print(model)
    
    torch.save(model, "/storage3/michel/Pretrained_NW/CIFAR10/init_model")
    
    
    help.test_model_accuracy("/storage3/michel/Pretrained_NW/CIFAR10/init_model", "CIFAR10")
