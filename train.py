import torch
from torch import optim
from torch.optim import Optimizer
import torch.nn as nn
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import ColorJitter
from torch.utils.tensorboard import SummaryWriter, writer

from tqdm import tqdm

from model import CNN


def train(model:nn.Module, dataloader: DataLoader, optimizer:Optimizer, criterion):
    model.train()
    running_loss = 0
    for x, labels in dataloader:
        x = x.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(train_loader)

    return train_loss

def valid(model:nn.Module, dataloader: DataLoader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, labels in dataloader:
            x = x.to(device)
            labels = labels.to(device)

            outputs = model(x)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            predicted = outputs.max(1, keepdim=True)[1]
            labels = labels.view_as(predicted)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(test_loader)
    val_acc = correct /total

    return val_loss, val_acc



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepareing MNIST dataset
    train_dataset = MNIST(root='./data', train=True, transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]), download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))

    #Limit label
    train_mask = (train_dataset.targets == 0) | (train_dataset.targets == 6)
    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = train_dataset.targets[train_mask]
    train_dataset.targets[train_dataset.targets == 6] = 1

    test_mask = (test_dataset.targets == 0) | (test_dataset.targets == 6)
    test_dataset.data = test_dataset.data[test_mask]
    test_dataset.targets = test_dataset.targets[test_mask]
    test_dataset.targets[test_dataset.targets == 6] = 1

    train_loader = DataLoader(dataset=train_dataset, batch_size = 128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size = 1, shuffle=False)


    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-5)

    writer = SummaryWriter(log_dir='./logs')
    writer.add_graph(model, torch.randn(128,1,32,32).to(device))

    epochs = 100

    for i in tqdm(range(epochs)):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = valid(model, test_loader, criterion)

        writer.add_scalar('train_loss', train_loss, i)
        writer.add_scalar('test_loss', val_loss, i)

    torch.save(model.state_dict(), './model/weights.pt')
    writer.close()

