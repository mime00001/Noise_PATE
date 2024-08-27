# Removing the necessity of public data in the PATE algorithm through noise distillation

Combining ideas from noise-engineered data-free Knowledge Distillation by Raikwar, Piyush and Mishra, Deepak (https://proceedings.neurips.cc/paper_files/paper/2022/file/1f96b24df4b06f5d68389845a9a13ed9-Paper-Conference.pdf)  with the PATE framework by Nicolas Papernot and Shuang Song and Ilya Mironov and Ananth Raghunathan and Kunal Talwar and Úlfar Erlingsson (https://arxiv.org/pdf/1802.08908). This code is the source code for my B.Sc. thesis "Removing the necessity of public data in the PATE algorithm through noise distillation".

## Abstract 

With the growing importance of machine learning in most aspects of society and technology, there are also rising concerns about the privacy implications of models trained on sensitive data. One promising approach to address these concerns is Private Aggregation of Teacher Ensembles, or PATE, which consist of an ensemble of "teachers", that transfers their knowledge to a "student" model. Intuitively this guarantees privacy, since the student does not have direct access to the training data. Furthermore the knowledge transfer is done in a noisy, privacy preserving manner. This method allows for strong privacy guarantees with high utility for sensitive data. This framework, requires available public data, to transfer knowledge from the teacher ensemble to the student model. 
While PATE tries to transfer knowledge from an ensemble of teacher models, there also exist approaches to transfer knowledge from a single large cumbersome model, to a smaller model. This technique is know as Knowledge Distillation (KD). Raikwar et al. have been working on a method to transfer knowledge from a large teacher model, to a student model, with the use of Gaussian noise. This repository combines their idea of using Gaussian noise for the model transfer, with the PATE framework.

## Dependencies

This codebase runs on `Python 3.12.3`. It uses common Python libraries, such as `torch` and `numpy`. All dependencies can be found in "requirements.txt".

## How to run

Running the main function on the MNIST dataset using Gaussian noise as transfer method can be done via running the noise_main.py file. First the teachers are trained on the private MNIST dataset, then the knowledge transfer to the student model is performed by using the BatchNorm trick and Gaussian noise. Training and querying the teachers will take a long time. When the teachers are trained once, the train_teachers variable can be set to False in the function call. 

All the plots can be created by running the function create_all_plots() in the plots.py file. Warning: this will take a long time.

## Results

```{=latex}
\begin{table}[ht]
\centering
\begin{tabular}{|c|c|c|c|c|}
\hline
Accuracy & $\varepsilon=5$ & $\varepsilon=8$ & $\varepsilon=10$ & $\varepsilon=20$ \\ \hline
PATE + RS (baseline)                 & 95.6\% $\pm$ 0.3\%    & 96.1\%* $\pm$ 0.5\%     & - & -  \\ \hline
PATE + CS                 & 95.9\% $\pm$ 0.5\%    & 97.2\%* $\pm$ 0.4\%    & -    & -    \\ \hline
Gaussian noise + RS              & 10.3\% $\pm$ 0.3\%      & 9.9\% $\pm$ 0.3\%     & 10.1\% $\pm$ 0.3\%      & 10.5\% $\pm$ 0.3\%     \\ \hline
Gaussian noise + CS              & 38\% $\pm$ 3.6\%  & 51.2\% $\pm$ 4\%    & 59.2\% $\pm$ 2.3\%      & 73.9\% $\pm$ 1.6\%    \\ \hline
FMNIST + RS                      & 29.5\% $\pm$ 1.9\%     & 33.9\% $\pm$ 1.9\%     & 34.4\% $\pm$ 1\%        & 42\% $\pm$ 4.3\%      \\ \hline
FMNIST + CS                      & 55.5\% $\pm$ 2.7\%     & 61.5\% $\pm$ 2.1\%     & 64.7\% $\pm$ 1.8\%         & 74.5\% $\pm$ 1.1\%     \\ \hline
\end{tabular}
\caption{The resulting accuracy for knowledge transfer using different datasets for MNIST. $T=150,\ \sigma_1=120,\ \sigma_2=40,\ \delta=10^{-5}$ for the public data and Gaussian noise. $T=200,\ \sigma_1=100,\ \sigma_2=20,\ \delta=10^{-5}$ for FMNIST~\cite{dataset_xiao2017fashionmnistnovelimagedataset}. \emph{*There is not enough public data to fulfill the whole privacy budget, $\varepsilon=6.47$}. Note that RS is running statistics and CS is current statistics.}
\end{table}
```

## References

PATE framework: https://github.com/tensorflow/privacy/tree/master/research/pate_2017 and https://github.com/tensorflow/privacy/tree/master/research/pate_2018

Gaussian KD: https://github.com/Piyush-555/GaussianDistillation/tree/main

Privacy accounting: https://github.com/cleverhans-lab/PrivatePrompts/tree/main/PromptPATE/pate