import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchsummary
import math
from torch.utils.tensorboard import SummaryWriter

from torch.optim import lr_scheduler
from model import ResNet
from data_loader import data_loader
from train import train
from test import test

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')
        
createFolder('./checkpoint/test')

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is applied.")
writer = SummaryWriter("./runs/ResNet_test")

### hyperparameter ###

batch_size = 128
num_epoch = 200
input_dim = 32

padding = 4
num_layers = 3
learning_rate = 0.1
friction = 0.9
lr_decay = 0.0001

train_loader, test_loader = data_loader(input_dim, padding, batch_size)
resnet = ResNet(num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=resnet.parameters(), lr=learning_rate, momentum=friction, weight_decay=lr_decay)

#decay_epoch = [math.ceil(num_epoch*0.5), math.ceil(num_epoch*0.75)]
decay_epoch = [100, 150]
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, gamma=0.1)

#checkpoint = torch.load('./checkpoint/checkpoint-26.pt')
#resnet.load_state_dict(checkpoint["model_state_dict"])
#optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#mid_epoch = checkpoint["epoch"]
checkpoint_num = 1

for epoch in range (num_epoch):
    print(f"epoch {epoch+1}\n-------------------------------")
    
    train(train_loader, resnet, criterion, optimizer, device, epoch)
    test(test_loader, resnet, criterion, device, epoch)
    step_lr_scheduler.step() 

    if (epoch+1) % 10 == 0:
        torch.save(
                {
                    "model": "resnet",
                    "epoch": epoch+1,
                    "model_state_dict": resnet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    },
                f"./checkpoint/test/checkpoint-{checkpoint_num}.pt",
                )
        checkpoint_num += 1

writer.flush()
writer.close()

print("Done!")

