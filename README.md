# DTEP-network-classification

DTEP: A Network Classification Method using Network Automata

# Intro
This paper presents a network classification method called DTEP that utilizes the concept of 
network automata to classify networks as a whole. The approach is described in detail in a recent 
publication (https://arxiv.org/abs/2211.13000), and this repository contains the code for executing 
the method. The repository allows for easy reproduction of the experiments conducted in the paper, 
which cover multiple databases.

# Methodology
The following image shows the general idea of the method:
![image](https://user-images.githubusercontent.com/36050575/226147932-6a2a7ec5-7324-47ea-bf15-a0e3f72f7b84.png)

In this study, we propose a method for network classification that utilizes Life-like Network Automata (LLNA). 
Specifically, each network in a given dataset is transformed into an LLNA, where the nodes of the network are 
evolved using life-like rules, such as those found in Conway's Game of Life. This process generates Density 
Time Evolution Patterns (DTEPs) that are then extracted from the LLNA. The frequency distributions of these 
DTEPs are used to derive features that are specific to each network. Finally, a Support Vector Machine (SVM) 
classification algorithm is employed to classify the different types of networks based on these features. Overall, 
our method offers a novel approach to network classification that takes advantage of LLNA and DTEPs to identify 
distinctive features of individual networks.

# Databases
The synthetic database can be found at ([Zenodo-synthetic-database])(https://zenodo.org/record/7749500#.ZByh0MLMJD8).
The real database is also available under ([Zenodo-real-database])(https://zenodo.org/record/7749500#.ZByh0MLMJD8)

# How to use the repository
After downloading the repository, in datasets.py, change the following variables for your own local path in which you downloaded the databases and want to save the results:
```
DATASETS_PATH = "your_dataset_path/"
TEP_OUTPUT_PATH = "result_path/TEP/"
FEATURE_OUTPUT_PATH = "result_path/features/"
```
