import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

# get dataloader
class Messidor_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.RandomRotation(15),
                                             transforms.RandomResizedCrop(size=512, scale=(0.9, 1)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(np.uint8(x))
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class Breast_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.RandomRotation(15),
                                             transforms.RandomResizedCrop(size=512, scale=(0.9, 1)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(np.uint8(x))
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class Breast_multi_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.RandomRotation(15),
                                             transforms.RandomResizedCrop(size=512, scale=(0.9, 1)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(np.uint8(x))
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


