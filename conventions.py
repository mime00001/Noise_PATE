"""
Naming conventions, some specs, hyperparameters and stuff
"""

#this code is taken from https://github.com/Piyush-555/GaussianDistillation/tree/main

def resolve_dataset(dataset_name):
    # That can't be defined here
    
    print(dataset_name)
    if "noise_" in dataset_name:
        dataset_name = dataset_name.replace("noise_", "")
    experiment_config = {
        'dataset_name': dataset_name,
    }
    if dataset_name=='CIFAR10':
        experiment_config['code_dim'] = 10
        experiment_config['model_teacher'] = "resnet18"
        experiment_config['model_student'] = "resnet18"
        experiment_config['inputs'] = 3
        experiment_config['channels'] = 3
        experiment_config['batch_size'] = 128
        experiment_config['BN_trick'] = True
    elif dataset_name=="MNIST":
        experiment_config['code_dim'] = 10
        experiment_config['model_teacher'] = "resnet18_mnist"
        experiment_config['model_student'] = "resnet18_mnist"
        experiment_config['inputs'] = 1
        experiment_config['channels'] = 1
        experiment_config['batch_size'] = 256
        experiment_config['BN_trick'] = True
    elif dataset_name=="TissueMNIST":
        experiment_config['code_dim'] = 8
        experiment_config['model_teacher'] = "resnet18_mnist"
        experiment_config['model_student'] = "resnet18_mnist"
        experiment_config['inputs'] = 1
        experiment_config['channels'] = 1
        experiment_config['batch_size'] = 256
        experiment_config['BN_trick'] = True
    elif dataset_name=="FMNIST":
        experiment_config['code_dim'] = 10
        experiment_config['model_teacher'] = "resnet18_mnist"
        experiment_config['model_student'] = "resnet18_mnist"
        experiment_config['inputs'] = 1
        experiment_config['channels'] = 1
        experiment_config['batch_size'] = 256
        experiment_config['BN_trick'] = True
    
    return experiment_config



def resolve_teacher_name(experiment_config):
    model_name = "teacher_"
    model_name += "{}_{}".format(
        experiment_config['dataset_name'],
        experiment_config['model_teacher']
    )
    model_name += ".model"
    return model_name

def resolve_student_name(experiment_config):
    model_name = "student_"
    model_name += "{}_{}".format(
        experiment_config['dataset_name'],
        experiment_config['model_student']
    )
    model_name += ".model"
    return model_name
