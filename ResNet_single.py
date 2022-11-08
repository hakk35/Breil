#!/usr/bin/python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./runs/ResNet_single")
import torchsummary

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is applied.")
 
### Hyperparameters ###
batch_size = 128
epochs = 200
input_dim = 32

friction = 0.9
learning_rate = 0.1
lr_decay = 0.0001

### number of layers ###
num_layers = 3

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')
        
createFolder('./checkpoint')

print('**************************************')
print('******* Used hyper-paramerters *******')
print(f'learning_rate       = {learning_rate}')
print(f'friction_coeff      = {friction}')
print(f'epochs              = {epochs}')
print('**************************************')

### Data Setting ###
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform_train = transforms.Compose([
    transforms.RandomCrop(input_dim, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
transform_test = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

class conv_block(nn.Module):
    def __init__(self, in_depth, out_depth, **kwargs):
        super(conv_block, self).__init__()

        self.conv = nn.Conv2d(in_depth, out_depth, kernel_size=3, padding=1, bias=False, **kwargs)
        self.batch = nn.BatchNorm2d(out_depth)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.conv(x)
        output = self.batch(output)
        output = self.relu(output)
        return output

class residual_block(nn.Module):
    def __init__(self, in_depth, out_depth, stride=1, **kwargs):
        super(residual_block, self).__init__()
        self.stride= stride
        self.conv1 = conv_block(in_depth, out_depth, stride=stride)
        self.conv2 = nn.Conv2d(out_depth, out_depth, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.batch = nn.BatchNorm2d(out_depth)

        if stride == 2:
            self.dim_reduc_conv= nn.Conv2d(in_depth, out_depth, kernel_size=1, stride=stride, bias=False, **kwargs)
            self.dim_reduc_batch = nn.BatchNorm2d(out_depth)

    def forward(self, x):
        output = self.conv1(x)
        output = self.batch(self.conv2(output))
        if self.stride == 2:
            output += self.dim_reduc_batch(self.dim_reduc_conv(x))
        else:
            output += x
        output = F.relu(output)
        return output

class get_layers(nn.Module):
    def __init__(self, in_depth, out_depth, stride, num_layers):
        super(get_layers, self).__init__()
        count = 0
        self.layers = nn.Sequential(residual_block(in_depth, out_depth, stride=stride))
        while count < num_layers-1:
            self.layers.append(residual_block(out_depth, out_depth, stride=1))
            count += 1

    def forward(self, x):
        output = self.layers(x)
        return output

class ResNet(nn.Module):
    def __init__(self, num_layers):
        super(ResNet, self).__init__()
        self.conv1 = conv_block(3, 16)

        self.layers_16 = get_layers(16,16, stride=1, num_layers=num_layers)
        self.layers_32 = get_layers(16,32, stride=2, num_layers=num_layers)
        self.layers_64 = get_layers(32,64, stride=2, num_layers=num_layers)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 10)

        ## weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = self.conv1(x)

        output = self.layers_16(output)
        output = self.layers_32(output)
        output = self.layers_64(output)

        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

resnet = ResNet(num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=resnet.parameters(), lr=learning_rate, momentum=friction, weight_decay=lr_decay)
decay_epoch = [100, 150]
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, gamma=0.1)
torchsummary.summary(resnet, (3,32,32))

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
    writer.add_scalar("Error/train", 100.0 - accuracy, epoch)
    print('train epoch : {} [{}/{}]| loss: {:.3f} | accuracy: {:.3f}'.format(epoch, batch_idx, len(train_loader), train_loss/(batch_idx+1), accuracy))

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
    writer.add_scalar("Error/test", 100.0 - accuracy, epoch)
    print('test epoch : {} [{}/{}]| loss: {:.3f} | accuracy: {:.3f}'.format(epoch, batch_idx, len(test_loader), test_loss/(batch_idx+1), accuracy))

#checkpoint = torch.load('./checkpoint/checkpoint-174.pt')
#resnet.load_state_dict(checkpoint["model_state_dict"])
#optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#mid_epoch = checkpoint["epoch"]
#checkpoint_num = 175
checkpoint_num = 1

for epoch in range (epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")

    train(train_loader, resnet, criterion, optimizer, device, epoch)
    test(test_loader, resnet, criterion, device, epoch)
    step_lr_scheduler.step()

    if (epoch+1) % 50 == 0:
        torch.save(
                {
                    "model": "resnet",
                    "epoch": epoch,
                    "model_state_dict": resnet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    },
                f"./checkpoint/checkpoint-{checkpoint_num}.pt",
                )
        checkpoint_num += 1

writer.flush()
writer.close()

print("Done!")

