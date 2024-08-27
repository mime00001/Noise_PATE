# PATE with noise-engineered data-free KD

Combining ideas from noise-engineered data-free Knowledge Distillation by Raikwar, Piyush and Mishra, Deepak (https://proceedings.neurips.cc/paper_files/paper/2022/file/1f96b24df4b06f5d68389845a9a13ed9-Paper-Conference.pdf)  with the PATE framework by Nicolas Papernot and Shuang Song and Ilya Mironov and Ananth Raghunathan and Kunal Talwar and Úlfar Erlingsson (https://arxiv.org/pdf/1802.08908).

## Abstract 

With the growing importance of machine learning in most aspects of society and technology, there are also rising concerns about the privacy implications of models trained on sensitive data. One promising approach to address these concerns is Private Aggregation of Teacher Ensembles, or PATE, which consist of an ensemble of "teachers", that transfers their knowledge to a "student" model. Intuitively this guarantees privacy, since the student does not have direct access to the training data. Furthermore the knowledge transfer is done in a noisy, privacy preserving manner. This method allows for strong privacy guarantees with high utility for sensitive data. This framework, requires available public data, to transfer knowledge from the teacher ensemble to the student model. 
While PATE tries to transfer knowledge from an ensemble of teacher models, there also exist approaches to transfer knowledge from a single large cumbersome model, to a smaller model. This technique is know as Knowledge Distillation (KD). Raikwar et al. have been working on a method to transfer knowledge from a large teacher model, to a student model, with the use of Gaussian noise. This repository combines their idea of using Gaussian noise for the model transfer, with the PATE framework.

## Dependencies

This codebase runs on `Python 3.12.3`. It uses common Python libraries, such as `torch` and `numpy`. All dependencies can be found in "requirements.txt".

## How to run

Running the main function on the MNIST dataset using Gaussian noise as transfer method can be done via running the noise_main.py file. First the teachers are trained on the private MNIST dataset, then the knowledge transfer to the student model is performed by using the BatchNorm trick and Gaussian noise. Training and querying the teachers will take a long time. When the teachers are trained once, the train_teachers variable can be set to False in the function call. 

All the plots can be created by running the function create_all_plots() in the plots.py file. Warning: this will take a long time.

## References

PATE framework: https://github.com/tensorflow/privacy/tree/master/research/pate_2017 and https://github.com/tensorflow/privacy/tree/master/research/pate_2018

Gaussian KD: https://github.com/Piyush-555/GaussianDistillation/tree/main

Privacy accounting: https://github.com/cleverhans-lab/PrivatePrompts/tree/main/PromptPATE/pate