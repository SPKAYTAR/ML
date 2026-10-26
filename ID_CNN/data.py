from torch.utils.data import DataLoader #dataset management, data batches etc
import torchvision.datasets as datasets #helps import datasets, 
import torchvision.transforms as transforms #dataset transformations

def CIFAR_TRAIN_DATA(batch_size):
    train_dataset = datasets.CIFAR10(root = 'dataset/', train = True, transform = transforms.ToTensor(),download = True )
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True) #auto divides into the batch_sizes
    return train_loader

def CIFAR_TEST_DATA(batch_size):
    test_dataset = datasets.CIFAR10(root = 'dataset/', train = False, transform = transforms.ToTensor(),download = True )
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=True) #auto divides into the batch_sizes
    return test_loader