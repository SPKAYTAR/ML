import torch
from tqdm import tqdm
import torch.nn as nn  # Neural network layers
from torchvision.models import resnet18  # Prebuilt ResNet-18 architecture
import time

def CHECK_ACCURACY(loader,model,device):
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
    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples )*100:.2f}')

    model.train()

def BENCHMARK(model, test_loader, device):
    model.eval()  # Evaluation mode

    num_correct = 0
    num_samples = 0
    total_time = 0

    # Count model parameters
    params = sum(p.numel() for p in model.parameters())

    # Calculate model weight size in MB
    size_mb = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / (1024 ** 2)

    with torch.inference_mode():

        # Warm up GPU using first batch
        if device.type == "cuda":
            x, _ = next(iter(test_loader))
            x = x.to(device)

            for _ in range(10):
                model(x)

            torch.cuda.synchronize()

        # Test model on entire dataset
        for x, y in tqdm(test_loader, desc="Benchmarking"):
            x = x.to(device)
            y = y.to(device)

            # Wait for GPU before timing
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            scores = model(x)  # Forward propagation

            # Wait for GPU to finish inference
            if device.type == "cuda":
                torch.cuda.synchronize()

            total_time += time.perf_counter() - start

            # Calculate accuracy
            predictions = scores.argmax(dim=1)
            num_correct += (predictions == y).sum().item()
            num_samples += y.size(0)

    # Calculate final metrics
    accuracy = 100 * num_correct / num_samples
    latency = total_time * 1000 / num_samples
    throughput = num_samples / total_time

    print("\n===== MODEL BENCHMARK =====")
    print(f"Accuracy:        {accuracy:.2f}%")
    print(f"Parameters:      {params:,}")
    print(f"Weight size:     {size_mb:.2f} MiB")
    print(f"Inference time:  {total_time:.3f} seconds")
    print(f"Latency:         {latency:.3f} ms/image")
    print(f"Throughput:      {throughput:.0f} images/sec")

    model.train()  # Restore training mode
