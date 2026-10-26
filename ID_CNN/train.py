import torch
from tqdm import tqdm
import time

def TRAIN_CNN_MODEL(num_epochs,train_loader,model,device, criterion, optimizer):
    warmup_gpu_CIFAR(model,device)
    start = time.perf_counter()
    for epoch in range(num_epochs):
        for batch_idx,(data,targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            #progress bar
        
            #move data batch to gpu
            data = data.to(device = device) 
            targets = targets.to(device=device)

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
    torch.cuda.synchronize() if device.type == "cuda" else None
    total_time = time.perf_counter() - start
    print(f"Total training time: {total_time:.2f} seconds")

def TRAIN_NN_MODEL(num_epochs,train_loader,model,device, criterion, optimizer):
    start = time.perf_counter()
    for epoch in range(num_epochs):
        for batch_idx,(data,targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            
        
            #move data batch to gpu
            data = data.to(device = device) 
            targets = targets.to(device=device)

            data=data.reshape(data.shape[0],-1) #flattens data 

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
    
    total_time = time.perf_counter() - start
    print(f"Total training time: {total_time:.2f} seconds")

def warmup_gpu_CIFAR(model, device):
    if device.type != "cuda":
        return

    model.eval()  # Evaluation mode
    dummy = torch.randn(64, 3, 32, 32, device=device)  

    with torch.no_grad():  
        for _ in range(10):
            model(dummy)  

    torch.cuda.synchronize() 
    model.train()  