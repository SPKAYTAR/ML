# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader  # Creates batches

from models import CNN1, SAVE_MODEL
from train import TRAIN_CNN_MODEL
from data import CIFAR_TRAIN_DATA, CIFAR_TEST_DATA
from test import BENCHMARK,CHECK_ACCURACY


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Load CIFAR-10 training and testing batches
train_loader = CIFAR_TRAIN_DATA(batch_size)
test_loader = CIFAR_TEST_DATA(batch_size)

# Initialize CNN on GPU
model = CNN1(in_channels=3, num_classes=10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train CNN
for runs in range(num_epochs):
    TRAIN_CNN_MODEL(
    1,
    train_loader,
    model,
    device,
    criterion,
    optimizer
    )
    CHECK_ACCURACY(test_loader,model,device)

# Benchmark trained model
BENCHMARK(model, test_loader, device)

# Save trained weights
SAVE_MODEL(model, "CIFAR10_CNN1")
