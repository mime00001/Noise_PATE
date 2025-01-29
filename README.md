# DIET-PATE: Knowledge Transfer in PATE without Public Data

## Abstract
The PATE algorithm is one of the canonical approaches to private machine learning, leveraging a private dataset to label a public dataset for knowledge transfer under differential privacy guarantees. However, its reliance on public data from the same distribution as the private data for the knowledge transfer poses a significant challenge, especially in medical or financial domains where such public data is usually unavailable. Recent advances have proposed using programmatically generated data as a substitute for public data in differentially private machine learning, opening new avenues for private knowledge transfer. In this work, we adapt the PATE framework to operate on programmatically generated data, removing the dependence on distribution-matched public data. The key to the success is to leverage the data-free knowledge distillation, which eliminates the shift in the distribution of hidden layer activations caused by feeding data from a different distribution than the training set to the ensemble of teacher models. We also demonstrate that the pretraining of teachers and the student on programmatically generated data increases substantially the overall performance by eliminating the need to learn generic features from scratch using the private data. Additionally, our method seamlessly extends to a distributed setting, where each teacher is trained by a different party. Through extensive benchmarking we show that our approach eliminates the reliance on public data and significantly outperforms the standard PATE method.


## Preparation

To run this code, you will need to generate the synthetic datasets Shaders21k (https://github.com/mbaradad/shaders21k/tree/main), FractalDB (https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch) StyleGAN-oriented and Dead Leaves (https://github.com/mbaradad/learning_with_noise).
They have to be stored in LOG_DIR_DATA, which you have to set manually. The data has to be stored in subfolders: "/shaders21k/", "/FractalDB/", "/stylegan-oriented/" and "/dead_leaves-mixed/" respectively.  

Further you will have to use the SimCLR framework (https://github.com/sthalles/SimCLR) to pretrain ResNet18 on the synthetic data. The changes to the SimCLR framework to fit to the task at hand are stored in SimCLR_changes. Please consult the README.md located in the folder to start the backbone training. 

Finally in the diet_pate.py file execute the ```python datasets.prepare_datasets_for_DIET_PATE() ``` function. This prepares all the data for the knowledge transfer.

## Executing DIET-PATE

The simplest way to start with the experiments is to first perform a full training run which trains all the teachers and a first student model. This can also be used to test if the datasets are stored correctly and the data directories have the correct naming across all files.

In the diet_pate.py file execute the full_run() function

```python full_run(target_dataset="MNIST", transfer_dataset="MNIST", backbone_name="stylegan", nb_teachers=200, SSL_teachers=True, train_teachers=True, compare=False, epsilon=5,  BN_trick=True) ```

and 

```python full_run(target_dataset="MNIST", transfer_dataset="MNIST", backbone_name=None, nb_teachers=200, SSL_teachers=False, train_teachers=True, compare=False, epsilon=5,  BN_trick=False) ```

You can then repeat the same for CIFAR10 by switching target_dataset and transfer_dataset to CIFAR10. For CIFAR10 use 50 teachers and for TissueMNIST 250. 

For the different methods, you will have to specify the variables as follows:

PATE : ```python backbone_name=None, SSL_teachers=False, BN_trick=False ```
DataFreeKD: ```python backbone_name=None, SSL_teachers=False, BN_trick=True ```
Pretraining: ```python backbone_name=stylegan, SSL_teachers=True, BN_trick=False ```
DIET-PATE: ```python backbone_name=stylegan, SSL_teachers=True, BN_trick=True ```


## Experimental results

The file plots.py was used to generate the main results for the plots in the paper.

The experiments for the CaPC results can be found in capc.py 