import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

def data_loader(input_dim, padding, batch_size):

    ### Transform Setting ###

    transform_train = transforms.Compose([
        transforms.RandomCrop(input_dim, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    ### Data Setting ###

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    ### Data Loader Setting ###

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

