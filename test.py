import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./runs/ResNet_test")

def test(test_loader, model, criterion, device, epoch):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            pred = model(inputs)
            
            test_loss += criterion(pred, labels).item()
            total += labels.size(0)
            correct += (pred.argmax(1) == labels).sum().item()

    accuracy = 100*correct/total
    writer.add_scalar("Error/test", 100.0 - accuracy, epoch+1)
    print('test epoch : {} [{}]| loss: {:.3f} | accuracy: {:.3f}'.format(epoch+1, len(test_loader), test_loss/(batch_idx+1), accuracy))


