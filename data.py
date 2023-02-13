from pathlib import Path
from typing import Any, Dict
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
import os
from torch.utils.data import Dataset
import numpy as np
def load_data_IID(config: Dict[str, Any]):
    n_clients = config["n_clients"]
    batch_size = config["batch_size"]
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    training_set = torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    testing_set = torchvision.datasets.MNIST('../data', train=False,
                       transform=transform)
    

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(training_set) // n_clients
    lengths = [partition_size] * n_clients
    datasets = random_split(training_set, lengths, torch.Generator().manual_seed(0))

    # If batch size equals 'inf' set the batch size to the training data size
    training_size = len(training_set)
    # testing_size = len(testing_set)
    if batch_size == "inf":
        batch_size = training_size
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 0 % validation set
        len_val = 0
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(0))
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    testloader = DataLoader(testing_set, batch_size=batch_size)

    return trainloaders, valloaders, testloader



class CustomImageDataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.img_labels = labels
        self.data = data
        self.transform=transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image.numpy().astype(np.uint8))
        return image, label

def load_data_nonIID(config: Dict[str, Any]):
    n_clients = config["n_clients"]
    batch_size = config["batch_size"]
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    training_set = torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    testing_set = torchvision.datasets.MNIST('../data', train=False,
                       transform=transform)
    
    # If batch size equals 'inf' set the batch size to the training data size
    training_size = len(training_set)
    # testing_size = len(testing_set)
    if batch_size == "inf":
        batch_size = training_size
    # sort the training set based on labels
    sorted_indices = torch.argsort(torch.Tensor(training_set.targets))
    sorted_training_data = training_set.data[sorted_indices]
    sorted_training_labels = torch.Tensor(training_set.targets)[sorted_indices]

    # training_set.data = sorted_training_data
    # training_set.targets = sorted_training_labels

    # devide into 200 shards each sherd will have 300 samples for MNIST
    shard_size = 300
    # devide the data into sherds
    shard_inputs = list(torch.split(torch.Tensor(sorted_training_data), shard_size))
    shard_labels = list(torch.split(torch.Tensor(sorted_training_labels), shard_size))
    
    shard_inputs_sorted, shard_labels_sorted = [], []
    sherd_per_class = 200 // 10
    # each class has about 20 sherds, we sort sherds based on class such that we have class 0 then 1 and so on....
    for i in range(sherd_per_class):
        for j in range(0, 200, sherd_per_class):
            shard_inputs_sorted.append(shard_inputs[i + j])
            shard_labels_sorted.append(shard_labels[i + j])

    # Now we store eachc clients data in dataset list
    datasets = []
    for i in range(0, len(shard_inputs_sorted), 2):
        # for each client
        train_data = torch.cat(shard_inputs_sorted[i:i + 2])
        train_targets = torch.cat(shard_labels_sorted[i:i + 2])
        datasets.append(CustomImageDataset(train_data, train_targets.long(),transform))
    # Split each partition into train/val and create DataLoader [to create the same results as the original paper we have set val size to 0]
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  
        len_val = 0 # 0 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(0))
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    testloader = DataLoader(testing_set, batch_size=batch_size)
    return trainloaders, valloaders, testloader