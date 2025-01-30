# DIET-PATE: Knowledge Transfer in PATE without Public Data

## Abstract
The PATE algorithm is one of the canonical approaches to private machine learning. It leverages a private dataset to label a public dataset for knowledge transfer from teachers to a student under differential privacy guarantees. However, PATE's reliance on public data from the same distribution as the private data for the knowledge transfer prevents its application, especially in medical and financial domains where such public data is usually unavailable. We find a solution to the problem by discovering a synergy between programmatically generated data and data free knowledge distillation. By pretraining teachers and the student on programmatically generated data, we increase substantially the overall performance by eliminating the need to learn generic features from scratch using the private data. We also leverage the data free knowledge distillation to eliminate the shift in the distribution of hidden layer activations caused by feeding the programmatically generated data from a different distribution than the private training set to the ensemble of teacher models. Additionally, our method seamlessly extends to a distributed setting, where each teacher is trained by a different party. Through extensive benchmarking we show that our approach eliminates the reliance on public data in the standard PATE method and its distributed derivative.


## Preparation

### Installing requirements

The code was run in python 3.12.4, the requirements can be installed with ``` pip install -r requirements.txt```

### Setting up the data and models

To run this code, you will need to generate the synthetic datasets Shaders21k (https://github.com/mbaradad/shaders21k/tree/main), FractalDB (https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch) StyleGAN-oriented and Dead Leaves (https://github.com/mbaradad/learning_with_noise).
They have to be stored in LOG_DIR_DATA, which you have to set manually. The data has to be stored in subfolders: "/shaders21k/", "/FractalDB/", "/stylegan-oriented/" and "/dead_leaves-mixed/" respectively.  

Further you will have to use the SimCLR framework (https://github.com/sthalles/SimCLR) to pretrain ResNet18 on the synthetic data. The changes to the SimCLR framework to fit to the task at hand are stored in SimCLR_changes. Please consult the README.md located in the folder to start the backbone training. 

Finally in the diet_pate.py file execute the ```datasets.prepare_datasets_for_DIET_PATE() ``` function. This prepares all the data for the knowledge transfer.

## Executing DIET-PATE

The simplest way to start with the experiments is to first perform a full training run which trains all the teachers and a first student model. This can also be used to test if the datasets are stored correctly and the data directories have the correct naming across all files.

In the diet_pate.py file execute the full_run() function

```full_run(target_dataset="MNIST", transfer_dataset="MNIST", backbone_name="stylegan", nb_teachers=200, SSL_teachers=True, train_teachers=True, compare=False, epsilon=5,  BN_trick=True) ```

and 

```full_run(target_dataset="MNIST", transfer_dataset="MNIST", backbone_name=None, nb_teachers=200, SSL_teachers=False, train_teachers=True, compare=False, epsilon=5,  BN_trick=False) ```

You can then repeat the same for CIFAR10 by switching target_dataset and transfer_dataset to CIFAR10. For CIFAR10 use 50 teachers and for TissueMNIST 250. 

For the different methods, you will have to specify the variables as follows:

PATE : ```backbone_name=None, SSL_teachers=False, BN_trick=False ```  
DataFreeKD: ```backbone_name=None, SSL_teachers=False, BN_trick=True ```  
Pretraining: ```backbone_name=stylegan, SSL_teachers=True, BN_trick=False ```  
DIET-PATE: ```backbone_name=stylegan, SSL_teachers=True, BN_trick=True ```  

The function ```pate_data.query_teachers()``` is used to query the teachers, either with DataFreeKD (BN_trick=True) or with DIET-PATE (BN_trick=True, SSL=True). The answeres from the teachers are then stored at /vote_array/transfer_dataset.npy.  

The vote array is then given as input to the PATE privacy accounting via ```pate_main.inference_pate()``` and the final labels are stored at /teacher_labels/transfer_dataset.npy.  

Finally the answered queries are used to train a student model.


## Experimental results

The file plots.py was used to generate the main results for the plots in the paper.

The experiments for the CaPC results can be found in capc.py 