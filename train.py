import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./runs/ResNet_test")

def train(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    accuracy = 100*correct/total
    writer.add_scalar("Error/train", 100.0 - accuracy, epoch+1)
    print('train epoch : {} [{}]| loss: {:.3f} | accuracy: {:.3f}'.format(epoch+1, len(train_loader), train_loss/(batch_idx+1), accuracy))


