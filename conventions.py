"""
Naming conventions, some specs, hyperparameters and stuff
"""

def resolve_dataset(dataset_name):
    # That can't be defined here
    experiment_config = {
        'dataset_name': dataset_name,
    }
    if dataset_name=='CIFAR10':
        experiment_config['code_dim'] = 10
        experiment_config['model_teacher'] = "resnet12"
        experiment_config['model_student'] = "resnet12"
        experiment_config['inputs'] = 3
        experiment_config['channels'] = 3
        experiment_config['batch_size'] = 256
    elif dataset_name=="MNIST":
        experiment_config['code_dim'] = 10
        experiment_config['model_teacher'] = "mnistresnet"
        experiment_config['model_student'] = "mnistresnet"
        experiment_config['inputs'] = 1
        experiment_config['channels'] = 1
        experiment_config['batch_size'] = 256
    return experiment_config

    #TODO MNIST


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
