## Introduction

This repository includes the code to reproduce my approach to the  [Kaggle challenge](https://www.kaggle.com/c/lish-moa) of classifying drugs based on their biological activity.


### The challenge
A Mechanism of Action (MoA) refers to a biochemical interaction through which a drug substance triggers a pharmacological effect. Target-identification and mechanism-of-action studies have important roles in small-molecule probe and drug discovery. 
The goal of the challenge is to predict multiple targets of the Mechanism of Action (MoA) response(s) of different samples, given various inputs such as gene expression data and cell viability data. More information can be found [here](https://www.kaggle.com/c/lish-moa). 



### The data
 Each sample in the data is treated with a specific drug, and we receive information on the dosage, time, and large variety of cell viability and gene expression related to the treatment.
Based on this data, we aim to create an algorithm that predicts the simulteanously predicts the probability of each of the MoA classes. 

There are 207 MoA outcomes that we aim to predict. Furthermore, there are also 403 MoA targets that do not necessarily need to be predicted (which I later refer to as *non scored targets*). These *non scored* targets may still be used to enhance the performace of the algorithm, but its outcomes are not known for the test data. 

##Solution overview

The approach in this repository is utilizing three deep learning models. It combines two neural feedforward neural networks with a 
[TabNet](https://arxiv.org/abs/1908.07442?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529) model. 
All models are implemented via PyTorch.

###Preprocessing

Because all used models are based on deep learning based and due to the nature of the data, there is not much feature engineering going on. Some preprocessing steps
that prepare the data before we feed it to the models are the following:

* Remove rows related to control group 
* Remove features with low variance 
* Construct principal components of original gene expression and cellular response data
* Generate row-wise descriptive statistics based on the cellular responses 

The principal components provide an uncorrelated denoised representation of the orginal data, that appear to provide a slight boost to the predictive power of the models.
Furthermore, it also helped to add some row-wise descriptive statistics (e.g. mean, kurtosis, skewness) of the distribution of the normalized gene expression and cell viability data. 

### Models

Two neural network model architectures are used: 

* A relatively simple three layer feedforward neural network.  

* A pre-trained five layer neural network. As mentioned, the data contains both scored and non-scored target columns.
In this model I apply transfer learning by first training my network on the non-scored targets. Afterwards, the weights of the last layers of the network are finetuned based the scored targets.
  
Both networks use batch normalization, dropout layers and non-linear activation function (leaky relu) between each layer. For training they both use early stopping based on a [MultilabelStratifiedKFold](https://github.com/trent-b/iterative-stratification) validation scheme, Adam for the optimization and [OneCycleLR](https://arxiv.org/abs/1708.07120) as a learning rate scheduler. [Label smoothing](https://arxiv.org/abs/1906.02629) is applied to mitigate the risk of overconfident predictions that are strongly penalized by the loss function that is used in the evaluation of the competition. 

* The last model is a PyTorch implementation of TabNet that is easily available [here](https://github.com/dreamquark-ai/tabnet). Some minor tweaks have been done to the hyperparameters of the model.


The final submission is an unweighted average of the predictions of the three models. 

### Reproducing the result

The result can be reproduced by running the following commands: 

`python train.py` \
`python inference.py`

The *train.py* script will construct, train and save the models. The *inference.py* script will load these models from the *trained_model* directory, and
generate a *submission.csv* file which contains the predictions for the test dataset. 

By default, the training procedure is done on GPU, so CUDA needs to be available as well as about 4GB of GPU memory. Dependencies are specified in the *requirement.txt* file. 

### References

* "Mechanisms of Action (MoA) Prediction" https://www.kaggle.com/c/lish-moa
* Original TabNet paper https://arxiv.org/abs/1908.07442
* PyTorch implementation of TabNet https://github.com/dreamquark-ai/tabnet


