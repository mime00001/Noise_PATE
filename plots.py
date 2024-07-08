import pate_main, pate_data
import experiments
import student
from utils import teachers, misc
import conventions
import datasets, models
import distill_gaussian

import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
import torchvision.transforms as transforms

LOG_DIR_DATA = "/disk2/michel/data"

def create_first_table():
    target_dataset = "MNIST"
    nb_teachers=200
    
    params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 5, "delta" : 1e-5}
    fmnist_params = {"threshold": 200, "sigma_threshold": 100, "sigma_gnmax": 20, "epsilon": 5, "delta" : 1e-5}
    
    vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=target_dataset, nb_teachers=nb_teachers)
    
    noise_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset="noise_MNIST", nb_teachers=nb_teachers)
    
    f_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset="FMNIST", nb_teachers=nb_teachers)
    
    #then perform inference PATE
    vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format("MNIST"))
    
    noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format("noise_MNIST"))
    
    f_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format("FMNIST"))
    
    
    vote_array=vote_array.T
    noise_vote_array = noise_vote_array.T
    f_vote_array = f_vote_array.T
    
    
    label_path = LOG_DIR_DATA + "/teacher_labels/MNIST.npy"
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format("noise_MNIST")
    fmnist_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format("FMNIST")
    epsilon_list = [5, 8, 10, 20]
    
    public_list=[[] for e in epsilon_list]
    gaussian_list=[[] for e in epsilon_list]
    FMNIST_list=[[] for e in epsilon_list]
    for j in range(5):
        for i, eps in enumerate(epsilon_list):
            #public data
            achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=label_path)
            num_answered = (pate_labels != -1).sum()
            final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=target_dataset, n_epochs=50)
            public_list[i].append((round(achieved_eps, 3), round(final_acc, 3), num_answered))
            
            #gaussian noise
            achieved_eps, pate_labels = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=noise_label_path)
            final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset="noise_MNIST", n_epochs=50)
            num_answered = (pate_labels != -1).sum()
            gaussian_list[i].append((round(achieved_eps, 3), round(final_acc, 3), num_answered))
            
            #fmnist
            achieved_eps, pate_labels = pate_main.inference_pate(vote_array=f_vote_array, threshold=fmnist_params["threshold"], sigma_threshold=fmnist_params["sigma_threshold"], sigma_gnmax=fmnist_params["sigma_gnmax"], epsilon=eps, delta=fmnist_params["delta"], num_classes=10, savepath=fmnist_label_path)
            final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset="FMNIST", n_epochs=50)
            num_answered = (pate_labels != -1).sum()
            FMNIST_list[i].append((round(achieved_eps, 3), round(final_acc, 3), num_answered))
        
        
    print("public data list")
    print(public_list)
    
    print("gaussian list")
    print(gaussian_list)
    
    print("fmnist data list")
    print(FMNIST_list)
    
    headers = ['eps=5', 'eps=8', "eps=10", "eps=20"]
    row_labels = [ "public_data", "Gaussian noise", "FMNIST data"]
    values = [
        [(np.mean(public_list[0], axis=0), np.std(public_list[0], axis=0)), (np.mean(public_list[1], axis=0), np.std(public_list[1], axis=0)), (np.mean(public_list[2], axis=0), np.std(public_list[2], axis=0)), (np.mean(public_list[3], axis=0), np.std(public_list[3], axis=0))],
        [(np.mean(gaussian_list[0], axis=0), np.std(gaussian_list[0], axis=0)), (np.mean(gaussian_list[1], axis=0), np.std(gaussian_list[1], axis=0)), (np.mean(gaussian_list[2], axis=0), np.std(gaussian_list[2], axis=0)), (np.mean(gaussian_list[3], axis=0), np.std(gaussian_list[3], axis=0))],
        [(np.mean(FMNIST_list[0], axis=0), np.std(FMNIST_list[0], axis=0)), (np.mean(FMNIST_list[1], axis=0), np.std(FMNIST_list[1], axis=0)), (np.mean(FMNIST_list[2], axis=0), np.std(FMNIST_list[2], axis=0)), (np.mean(FMNIST_list[3], axis=0), np.std(FMNIST_list[3], axis=0))]
    ]
    
    fig, ax = plt.subplots(figsize=(50, 30))

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=values, colLabels=headers, rowLabels=row_labels, loc='center', cellLoc='center')

    # Adjust layout
    plt.subplots_adjust(left=0.2, top=0.8)

    # Save the table to a file
    plt.savefig('table 1_std.png')


def create_second_table():
    
    
    params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 10, "delta" : 1e-5}
    
    
    teachers.util_train_teachers_same_init(dataset_name="MNIST", n_epochs=50, nb_teachers=200)
    
    
    noise_vote_array = pate_data.query_teachers(target_dataset="MNIST", query_dataset="noise_MNIST", nb_teachers=200)
    noise_vote_array = noise_vote_array.T
    
    #then perform inference pate
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format("noise_MNIST")
    noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
    
    
    #then train the student on Gaussian noise    
    sameT_diffS_acc = student.util_train_student(target_dataset="MNIST", transfer_dataset="noise_MNIST", n_epochs=60, lr=0.001, optimizer="Adam", kwargs=params, use_test_loader=True)
    sameT_sameS_acc = student.util_train_student_same_init(target_dataset="MNIST", transfer_dataset="noise_MNIST", n_epochs=60, use_test_loader=True)
    
    
    
    teachers.util_train_teachers(dataset_name="MNIST", n_epochs=50, nb_teachers=200)
    
    noise_vote_array = pate_data.query_teachers(target_dataset="MNIST", query_dataset="noise_MNIST", nb_teachers=200)
    noise_vote_array = noise_vote_array.T
    
    #then perform inference pate
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format("noise_MNIST")
    noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
    
    
    #then train the student on Gaussian noise    
    diffT_diffS_acc = student.util_train_student(target_dataset="MNIST", transfer_dataset="noise_MNIST", n_epochs=60, lr=0.001, optimizer="Adam", kwargs=params, use_test_loader=True)
    diffT_sameS_acc = student.util_train_student_same_init(target_dataset="MNIST", transfer_dataset="noise_MNIST", n_epochs=60, use_test_loader=True)
    
    
    headers = ['Same teacher init', 'Different teacher init']
    row_labels = [ "Same student init", "different student init"]
    values = [
        [sameT_sameS_acc, diffT_sameS_acc],
        [ sameT_diffS_acc, diffT_diffS_acc]
    ]
    
    fig, ax = plt.subplots()

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=values, colLabels=headers, rowLabels=row_labels, loc='center', cellLoc='center')

    # Adjust layout
    plt.subplots_adjust(left=0.2, top=0.8)

    # Save the table to a file
    plt.savefig('table 2.png')

def create_third_table():
    
    noise_vote_array = pate_data.query_teachers(target_dataset="MNIST", query_dataset="noise_MNIST", nb_teachers=200).T
    params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 10, "delta" : 1e-5}
    
    #then perform inference pate
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format("noise_MNIST")
    noise_votes = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=params["epsilon"], delta=params["delta"], num_classes=10, savepath=noise_label_path) 
    
    h=[]
    lo = []
    la = []
    s = []
    
    for i in range(5):
        histogram_acc = experiments.use_histogram()
        logits_acc = experiments.use_logits()
        label_acc = student.util_train_student(target_dataset="MNIST", transfer_dataset="noise_MNIST", n_epochs=60, lr=0.001, optimizer="Adam", use_test_loader=False, loss="xe", label=True)
        softmax_acc = experiments.use_softmax()
        
        h.append(histogram_acc)
        lo.append(logits_acc)
        la.append(label_acc)
        s.append(softmax_acc)



    values = [
        ["label_acc", "histogram_acc", "softmax_acc", "logits_acc"],
        [(np.mean(la), np.std(la)), (np.mean(h), np.std(h)), (np.mean(s), np.std(s)), (np.mean(lo), np.std(lo))]
    ]

    fig, ax = plt.subplots()

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=values, loc='center', cellLoc='center')

    # Adjust layout
    plt.subplots_adjust(left=0.2, top=0.8)

    # Save the table to a file
    plt.savefig('table 3.png')
    
def create_forth_table():
    
    params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 10, "delta" : 1e-5}
    
    num_datapoints = [0, 2000, 4000, 6000, 8000, 10000, 15000, 20000]
    
    
    accuracies = []
    
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("MNIST")
    
    transfer_loader, _, _ = eval("datasets.get_{}_student({})".format(
        "noise_MNIST",
        experiment_config["batch_size"]
    ))
    batch_size=256
    num_workers=4
    validation_size=0.2
    
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
    
    
    
    train_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    test_loader = torch.utils.data.DataLoader(partition_test, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    for n in num_datapoints:
        if not n==0:
            partition_valid = [trainset[i] for i in range(n)]
            
            valid_loader = torch.utils.data.DataLoader(partition_valid, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        else:
            valid_loader=None
            
        student_model = model = eval("models.{}.Target_Net({}, {})".format(
            experiment_config["model_student"],
            experiment_config["inputs"],
            experiment_config["code_dim"]
        )).to(device)
        
        metrics = student.train_student(student_model, transfer_loader, test_loader, 60, 0.001, 0, True, device, False,LOG_DIR="/disk2/michel/", optim="Adam", test_loader=valid_loader, loss="xe", label=True)
        
        acc=metrics[3][-1]
    
        accuracies.append(acc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(num_datapoints, accuracies, marker='o', linestyle='-')

    # Adding titles and labels
    plt.title('Accuracy depending on throughput before evaluation')
    plt.xlabel('Number of datapoints')
    plt.ylabel('Accuracy')

    # Displaying the plot
    plt.grid(True)
    plt.savefig('table 4.png')
    
    
def create_kd_data_plot():
    
    params = {"threshold": 150, "sigma_threshold": 120, "sigma_gnmax": 40, "epsilon": 10, "delta" : 1e-5}
    
    num_datapoints = [2048, 4096, 6144, 8192, 10240, 15000, 20000]
    
    
    accuracies = []
    
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("MNIST")
    
    batch_size=256
    num_workers=4
    validation_size=0.2
    LOG_DIR = "/disk2/michel/"
    
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
    
    metrics_list_label = []
    metrics_list_logits = []
    
    for n in num_datapoints:
        partition_train = [trainset[i] for i in range(n)]
        valid_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        
        
        teacher_name = conventions.resolve_teacher_name(experiment_config)
        teacher_path = os.path.join("/disk2/michel", "Pretrained_NW","{}".format("MNIST"), teacher_name)
        teacher_nw = torch.load(teacher_path)
        teacher_nw.to(device)

        student_nw = eval("models.{}.Target_Net({}, {})".format(
            experiment_config['model_student'],
            experiment_config['inputs'],
            experiment_config['code_dim']
        )).to(device)
        
        len_batch = len(valid_loader)
        
        m = distill_gaussian.distill_using_noise(None, teacher_nw, student_nw, valid_loader, 75, len_batch, 1e-3, True, device, False, LOG_DIR, label=True, test_loader=None)
        metrics_list_label.append(m[2][-1])
        
        m = distill_gaussian.distill_using_noise(None, teacher_nw, student_nw, valid_loader, 75, len_batch, 1e-3, True, device, False, LOG_DIR, label=False, test_loader=None)
        metrics_list_logits.append(m[2][-1])
    
    
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    len_batch=len(valid_loader)
        
    teacher_name = conventions.resolve_teacher_name(experiment_config)
    teacher_path = os.path.join("/disk2/michel", "Pretrained_NW","{}".format("MNIST"), teacher_name)
    teacher_nw = torch.load(teacher_path)
    teacher_nw.to(device)

    student_nw = eval("models.{}.Target_Net({}, {})".format(
        experiment_config['model_student'],
        experiment_config['inputs'],
        experiment_config['code_dim']
    )).to(device)

    
    
    base_line_logits = distill_gaussian.distill_using_noise(None, teacher_nw, student_nw, valid_loader, 75, len_batch, 1e-3, True, device, False, LOG_DIR, label=False, test_loader=None, different_noise=True)[2][-1]
    base_line_label = distill_gaussian.distill_using_noise(None, teacher_nw, student_nw, valid_loader, 75, len_batch, 1e-3, True, device, False, LOG_DIR, label=True, test_loader=None, different_noise=True)[2][-1]
    
    plt.ylim(0, 1)
    
    
    plt.plot(num_datapoints, metrics_list_label, label=f"Accuracy with label", color="tab:blue")
    plt.plot(num_datapoints, metrics_list_logits, label=f"Accuracy with logits", color="tab:orange")
    plt.axhline(base_line_label, color="b", linestyle="--", label="Baseline for labels")
    plt.axhline(base_line_logits, color="r", linestyle="--", label="Baseline for logits")
    
    plt.title('Knowledge Distillation with set number of samples')
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("plot 1.png")
    plt.close()
    
    
def consensus_plots_MNIST():
    
    #teachers.util_train_teachers("MNIST", 60, 200)
    
    #pate_data.query_teachers("MNIST", "noise_MNIST", 200)
    
    experiments.plot_count_histogram("consensus_diff_noiseMNIST.png", "/disk2/michel/data/vote_array/noise_MNIST.npy")
    
    teachers.util_train_teachers_same_init("MNIST", 75, 200)
    
    pate_data.query_teachers("MNIST", "noise_MNIST", 200)
    
    experiments.plot_count_histogram("consensus_same_noiseMNIST.png", "/disk2/michel/data/vote_array/noise_MNIST.npy")


def FMNIST_plot():
    params = {"threshold": 190, "sigma_threshold": 100, "sigma_gnmax": 50, "epsilon": 10, "delta" : 1e-6}
    
    np.set_printoptions(suppress=True)
    
    eps = [5, 8, 10, 20]
    
    FMNIST_list=[[] for e in eps]
    
    
    for i in range(5):
        for j, e in enumerate(eps):
            #f, n = only_transfer_set("MNIST", "FMNIST", epsilon=e)
            #FMNIST_list[j].append((f, n))
            break
    
    
    
    values = [
        [np.mean(FMNIST_list[0], axis=0), np.mean(FMNIST_list[1], axis=0), np.mean(FMNIST_list[2], axis=0), np.mean(FMNIST_list[3], axis=0)],
        [np.std(FMNIST_list[0], axis=0) , np.std(FMNIST_list[1], axis=0), np.std(FMNIST_list[2], axis=0), np.std(FMNIST_list[3], axis=0)]
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    headers = ['eps=5', 'eps=8', "eps=10", "eps=20"]
    row_labels = [ "mean", "std"]

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=values, colLabels=headers, rowLabels=row_labels, loc='center', cellLoc='center')

    # Adjust layout
    plt.subplots_adjust(left=0.2, top=0.8)

    # Save the table to a file
    plt.savefig('table 1_fmnist.png')
    #help.run_parameter_search("/vote_array/FMNIST.npy", "./pate_params_FMNIST")


def create_first_table_SVHN():
    target_dataset = "SVHN"
    nb_teachers=250
    
    np.set_printoptions(suppress=True)
    
    params = {"threshold": 150, "sigma_threshold": 100, "sigma_gnmax": 40, "epsilon": 5, "delta" : 1e-6}
    fmnist_params = {"threshold": 150, "sigma_threshold": 100, "sigma_gnmax": 40, "epsilon": 5, "delta" : 1e-6}
    
    vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset=target_dataset, nb_teachers=nb_teachers)
    
    noise_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset="noise_SVHN", nb_teachers=nb_teachers)
    
    f_vote_array = pate_data.query_teachers(target_dataset=target_dataset, query_dataset="CIFAR10", nb_teachers=nb_teachers)
    
    #then perform inference PATE
    vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format("SVHN"))
    
    noise_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format("noise_SVHN"))
    
    f_vote_array = np.load(LOG_DIR_DATA + "/vote_array/{}.npy".format("CIFAR10"))
    
    
    vote_array=vote_array.T
    noise_vote_array = noise_vote_array.T
    f_vote_array = f_vote_array.T
    
    
    label_path = LOG_DIR_DATA + "/teacher_labels/MNIST.npy"
    noise_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format("noise_SVHN")
    fmnist_label_path = LOG_DIR_DATA + "/teacher_labels/{}.npy".format("CIFAR10")
    epsilon_list = [5, 8, 10, 20]
    
    public_list=[[] for e in epsilon_list]
    gaussian_list=[[] for e in epsilon_list]
    FMNIST_list=[[] for e in epsilon_list]
    for j in range(5):
        for i, eps in enumerate(epsilon_list):
            #public data
            achieved_eps, pate_labels = pate_main.inference_pate(vote_array=vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=label_path)
            num_answered = (pate_labels != -1).sum()
            final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset=target_dataset, n_epochs=50)
            public_list[i].append((achieved_eps, final_acc, num_answered))
            
            #gaussian noise
            achieved_eps, pate_labels = pate_main.inference_pate(vote_array=noise_vote_array, threshold=params["threshold"], sigma_threshold=params["sigma_threshold"], sigma_gnmax=params["sigma_gnmax"], epsilon=eps, delta=params["delta"], num_classes=10, savepath=noise_label_path)
            final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset="noise_SVHN", n_epochs=50)
            num_answered = (pate_labels != -1).sum()
            gaussian_list[i].append((achieved_eps , final_acc, num_answered))
            
            #CIFAR10
            achieved_eps, pate_labels = pate_main.inference_pate(vote_array=f_vote_array, threshold=fmnist_params["threshold"], sigma_threshold=fmnist_params["sigma_threshold"], sigma_gnmax=fmnist_params["sigma_gnmax"], epsilon=eps, delta=fmnist_params["delta"], num_classes=10, savepath=fmnist_label_path)
            final_acc = student.util_train_student(target_dataset=target_dataset, transfer_dataset="CIFAR10", n_epochs=50)
            num_answered = (pate_labels != -1).sum()
            FMNIST_list[i].append((achieved_eps, final_acc, num_answered))
        
        
    print("public data list")
    print(public_list)
    
    print("gaussian list")
    print(gaussian_list)
    
    print("fmnist data list")
    print(FMNIST_list)
    
    headers = ['eps=5', 'eps=8', "eps=10", "eps=20"]
    row_labels = [ "public_data", "Gaussian noise", "CIFAR10 data"]
    values = [
        [np.mean(public_list[0], axis=0), np.mean(public_list[1], axis=0), np.mean(public_list[2], axis=0), np.mean(public_list[3], axis=0)],
        [np.mean(gaussian_list[0], axis=0), np.mean(gaussian_list[1], axis=0), np.mean(gaussian_list[2], axis=0) , np.mean(gaussian_list[3], axis=0)],
        [np.mean(FMNIST_list[0], axis=0), np.mean(FMNIST_list[1], axis=0), np.mean(FMNIST_list[2], axis=0), np.mean(FMNIST_list[3], axis=0)]
    ]
    
    fig, ax = plt.subplots(figsize=(30, 10))

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=values, colLabels=headers, rowLabels=row_labels, loc='center', cellLoc='center')

    # Adjust layout
    plt.subplots_adjust(left=0.2, top=0.8)

    # Save the table to a file
    plt.savefig('table 1_mean_SVHN.png')
    
    headers = ['eps=5', 'eps=8', "eps=10", "eps=20"]
    row_labels = [ "public_data", "Gaussian noise", "CIFAR10 data"]
    values = [
        [np.std(public_list[0], axis=0), np.std(public_list[1], axis=0), np.std(public_list[2], axis=0), np.std(public_list[3], axis=0)],
        [np.std(gaussian_list[0], axis=0), np.std(gaussian_list[1], axis=0), np.std(gaussian_list[2], axis=0) , np.std(gaussian_list[3], axis=0)],
        [np.std(FMNIST_list[0], axis=0), np.std(FMNIST_list[1], axis=0), np.std(FMNIST_list[2], axis=0), np.std(FMNIST_list[3], axis=0)]
    ]
    
    fig, ax = plt.subplots(figsize=(30, 10))

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(cellText=values, colLabels=headers, rowLabels=row_labels, loc='center', cellLoc='center')

    # Adjust layout
    plt.subplots_adjust(left=0.2, top=0.8)

    # Save the table to a file
    plt.savefig('table 1_std_SVHN.png')


def create_kd_data_plot_SVHN():
    
    params = {"threshold": 150, "sigma_threshold": 100, "sigma_gnmax": 40, "epsilon": 10, "delta" : 1e-6}
    
    num_datapoints = [2048, 4096, 6144, 8192, 10240, 15000, 20000, 40000, 60000, 100000]
    
    
    accuracies = []
    
    device = misc.get_device()
    experiment_config = conventions.resolve_dataset("SVHN")
    
    batch_size=256
    num_workers=4
    validation_size=0.2
    LOG_DIR = "/disk2/michel/"
    
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
    
    metrics_list_label = []
    metrics_list_logits = []
    
    for n in num_datapoints:
        partition_train = [trainset[i] for i in range(n)]
        valid_loader = torch.utils.data.DataLoader(partition_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        
        
        teacher_name = conventions.resolve_teacher_name(experiment_config)
        teacher_path = os.path.join("/disk2/michel", "Pretrained_NW","{}".format("SVHN"), teacher_name)
        teacher_nw = torch.load(teacher_path)
        teacher_nw.to(device)

        student_nw = eval("models.{}.Target_Net({}, {})".format(
            experiment_config['model_student'],
            experiment_config['inputs'],
            experiment_config['code_dim']
        )).to(device)
        
        len_batch = len(valid_loader)
        
        m = distill_gaussian.distill_using_noise(None, teacher_nw, student_nw, valid_loader, 75, len_batch, 1e-3, True, device, False, LOG_DIR, label=True, test_loader=None)
        metrics_list_label.append(m[2][-1])
        
        m = distill_gaussian.distill_using_noise(None, teacher_nw, student_nw, valid_loader, 75, len_batch, 1e-3, True, device, False, LOG_DIR, label=False, test_loader=None)
        metrics_list_logits.append(m[2][-1])
    
    
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    len_batch=len(valid_loader)
        
    teacher_name = conventions.resolve_teacher_name(experiment_config)
    teacher_path = os.path.join("/disk2/michel", "Pretrained_NW","{}".format("SVHN"), teacher_name)
    teacher_nw = torch.load(teacher_path)
    teacher_nw.to(device)

    student_nw = eval("models.{}.Target_Net({}, {})".format(
        experiment_config['model_student'],
        experiment_config['inputs'],
        experiment_config['code_dim']
    )).to(device)

    
    
    base_line_logits = distill_gaussian.distill_using_noise(None, teacher_nw, student_nw, valid_loader, 75, len_batch, 1e-3, True, device, False, LOG_DIR, label=False, test_loader=None, different_noise=True)[2][-1]
    base_line_label = distill_gaussian.distill_using_noise(None, teacher_nw, student_nw, valid_loader, 75, len_batch, 1e-3, True, device, False, LOG_DIR, label=True, test_loader=None, different_noise=True)[2][-1]
    
    plt.ylim(0, 1)
    
    
    plt.plot(num_datapoints, metrics_list_label, label=f"Accuracy with label", color="tab:blue")
    plt.plot(num_datapoints, metrics_list_logits, label=f"Accuracy with logits", color="tab:orange")
    plt.axhline(base_line_label, color="b", linestyle="--", label="Baseline for labels")
    plt.axhline(base_line_logits, color="r", linestyle="--", label="Baseline for logits")
    
    plt.title('Knowledge Distillation with set number of samples')
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("plot1_SVHN.png")
    plt.close()