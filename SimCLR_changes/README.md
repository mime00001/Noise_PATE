## About the SimCLR changes and setting up the synthetic pretraining

When the synthetic datasets are stored in their respective locations, you can pretrain a ResNet18 model, by executing the run.py file. E.g.:

python run.py --dataset-name="dead_leaves" -data="/data" --epochs 50 --single_channel

This train a ResNet18 model on the dead_leaves data stored in /data for 50 epochs.
The following datasets work to train a backbone:
stylegan, dead_leaves, shaders21k_grey, fractaldb, fmnist, shaders21k_rgb, dead_leaves_rgb, stylegan_rgb

Note that for our results we used stylegan and shaders21k_rgb. After the model has been pretrained, you need to store it at LOG_DIR + "Pretrained_NW/", such that it can be used further to work as backbone for the student and teacher models. 
You will also have to rename the models to backbone_{dataset_name}.pth.tar, e.g. backbone_stylegan.pth.tar