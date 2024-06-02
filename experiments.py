from utils import help, misc
import datasets, conventions

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
        print("True label: {}, Ensemble predictions: ".format(true_labels[i]), end="")
        output = ", ".join(f'{number}: {count}' for number, count in all_labels[i].items())
        print(output, end = "")
        print(" Predicted label by ensemble: {}".format(predicted_labels[i]))
        
        
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
        
    
plot_count_histogram()
    
    