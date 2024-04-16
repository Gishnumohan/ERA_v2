import torchvision
import torch.utils.data as data

""" Download CIFAR10 dataset to TRAIN and TEST"""
def getCIFAR10dataset(root="./data", train_flag=True, download_flag=True):
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=train_flag, download=download_flag
    )
    return dataset

CIFAR_CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CIFAR10_dataset(data.Dataset):
    def __init__(self, data, targets, transforms=None):
        self.data = data
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        X = self.data[item]
        y = self.targets[item]
        if self.transforms is not None:
            X = self.transforms(image=X)["image"]
        return X, y