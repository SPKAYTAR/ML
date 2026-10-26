#Imports
import torch
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F #relu, tanh etc
from torch.utils.data import DataLoader #dataset management, data batches etc
import torchvision.datasets as datasets #helps import datasets, 
import torchvision.transforms as transforms #dataset transformations
from tqdm import tqdm  # Training progress bar


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
class CNN(nn.Module):
    def __init__(self,in_channels=1,num_classes=10): #greyscale pictures
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,   # 1 grayscale input channel
            out_channels=8,  # 8 kernels
            kernel_size=(3,3),  # Each kernel is 3×3
            stride=(1,1),     # Move 1 pixel at a time
             padding=(1,1)     # Add 1 pixel of padding around the image
        )
        self.pool = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2))
        self.conv2  = nn.Conv2d(in_channels=8, out_channels=16,kernel_size=(3,3), stride=(1,1),padding=(1,1) )
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc1(x)

        return x



# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #uses gpu if available need nvidia
# hyperparameters
in_channel=1
input_size = 784
num_classes = 10
learning_rate = .001 #Wnew = Wold - (learning_rate)(dL/dW )
batch_size = 64
num_epochs = 5
#load data

#Load Data
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(),download = True )
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True) #auto divides into the batch_sizes

test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(),download = True )
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=True) #auto divides into the batch_sizes
#Initialize network
#model = NN(input_size = input_size,num_classes=num_classes).to(device) #creates the model 
model = CNN().to(device)

#loss and optimizer objects
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#train netowrk
for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        #progress bar
        
        #move data batch to gpu
        data = data.to(device = device) 
        targets = targets.to(device=device)

        #data=data.reshape(data.shape[0],-1) #flattens data ,  dont need flatenning for cnn

        #output of a run
        scores=model(data)
        #loss calculated from average cross entropy of batch
        loss=criterion(scores,targets ) # Calculates loss and returns a tensor linked to the backpropagation graph
        #clear old gradient
        optimizer.zero_grad()
        #Calculating backpropagation gradient stores in neuron
        loss.backward()
        #applies gradient to weights basedo n learning_rate
        optimizer.step()

# Save trained weights and biases to a file
#torch.save(model.state_dict(), "mnist_model.pth")\

# Copy saved weights and biases into the model
#weights = torch.load("mnist_model.pth",map_location=device,weights_only=True)
#model.load_state_dict(weights)
# check accuracy on trainig and test to see how good our model i 


def check_accuracy(loader,model):
    #progress bar
    num_correct = 0
    num_samples = 0
    #eval mode, doesnt care about gradient
    model.eval()
    #doesnt track gradients
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Testing"):
            #moves to gpu
            #image
            x = x.to(device=device) #x.shape = [64, 1, 28, 28]
            #correct label
            y = y.to(device=device)
            #flattens image to 784 instead of 28x28
            #x=x.reshape(x.shape[0],-1) dont need flattening in cnn

            #model output
            scores=model(x)
            #most activate output 
            predictions=scores.max(1).indices   #max index of value in each row
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
    print(f'Got {num_correct} / {num_samples} wiht accuracy {float(num_correct)/float(num_samples )*100:.2f}')

    model.train()
    

check_accuracy(train_loader,model)