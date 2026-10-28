# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader  # Creates batches
import torchvision.transforms as transforms #dataset transformations
from models import CNN1, SAVE_MODEL, LOAD_MODEL, CREATE_RESNET
from train import TRAIN_CNN_MODEL
from data import CIFAR_TRAIN_DATA, CIFAR_TEST_DATA
from test import BENCHMARK,CHECK_ACCURACY



# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("CUDA available:", torch.cuda.is_available())
print("Selected device:", device)
# Load CIFAR-10 training and testing batches
train_loader = CIFAR_TRAIN_DATA(batch_size)
train_loader.dataset.transform =transforms.Compose([
    transforms.ToTensor(),  # Convert pixels from 0-255 to 0-1
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])
test_loader = CIFAR_TEST_DATA(batch_size)
test_loader.dataset.transform = transforms.Compose([
    transforms.ToTensor(),  # Convert pixels from 0-255 to 0-1
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])



# Benchmark normal model
print("===== RESNET =====")







