#Imports
import torch
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F #relu, tanh etc
from torch.utils.data import DataLoader #dataset management, data batches etc
import torchvision.datasets as datasets #helps import datasets, 
import torchvision.transforms as transforms #dataset transformations



#Create fully connected network
class NN(nn.Module): #Class NN inherits from nn.module
    def __init__(self,input_size, num_classes): #initialization #input size = 28x28
        super().__init__() #initializing inheritance
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x): #forward propagation
        x = F.relu(self.fc1(x)) #passes x through layer 1 applies relu
        x = self.fc2(x) #passes relu output thorugh layer 2, outputs logits
        return x

#create CNN
class CNN1(nn.Module):
    def __init__(self,in_channels=1,num_classes=10,size=(32,32)): #greyscale pictures
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,   # 1 grayscale input channel
            out_channels=8,  # 8 kernels
            kernel_size=(3,3),  # Each kernel is 3×3
            stride=(1,1),     # Move 1 pixel at a time
             padding=(1,1)     # Add 1 pixel of padding around the image
        ) #input is 3x32x32
        self.pool = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2)) #input is 8x32x32
        self.conv2  = nn.Conv2d(in_channels=8, out_channels=16,kernel_size=(3,3), stride=(1,1),padding=(1,1) ) #16x16x16
        self.fc1 = nn.Linear(16*8*8, num_classes) #16x8x8

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc1(x)

        return x

class CNN2(nn.Module):
    def __init__(self,in_channels=1,num_classes=10,size=(32,32)): #greyscale pictures
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,   
            out_channels=16,  # 16 kernels
            kernel_size=(3,3),  # Each kernel is 3×3
            stride=(1,1),     # Move 1 pixel at a time
             padding=(0,0)     # No padding
        ) #input is 3x32x32
        self.pool = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2)) #input is 16x30x30
        self.conv2  = nn.Conv2d(in_channels=8, out_channels=16,kernel_size=(3,3), stride=(1,1),padding=(1,1) ) #16x16x16
        self.fc1 = nn.Linear(16*8*8, num_classes) #16x8x8

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc1(x)

        return x
#class CNN2(nn.Module): #handcrafted kernels
def SAVE_MODEL(model,name):
    torch.save(model.state_dict(), name + ".pth")

def LOAD_MODEL(model,name,device):
    model.load_state_dict(torch.load(name + ".pth", map_location=device, weights_only=True))

