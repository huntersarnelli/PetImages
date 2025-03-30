import os
import shutil
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


#Custom Dataset
class PetDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label




#Define CNN Model
class PetCNN(nn.Module):
    def __init__(self):
        super(PetCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
    


# Data Augmentation and Preprocessing
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2, scale=(0.7, 1.3)),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])







class AdvancedPetCNN(nn.Module):
    def __init__(self):
        super(AdvancedPetCNN, self).__init__()
        
        # Initial conv block
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Residual block 1
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.shortcut1 = nn.Conv2d(64, 128, 1)  # Shortcut for residual
        
        # Residual block 2
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.shortcut2 = nn.Conv2d(128, 256, 1)  # Shortcut for residual
        
        # Residual block 3
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.shortcut3 = nn.Conv2d(256, 512, 1)  # Shortcut for residual
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        # Initial conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Residual block 1
        identity = self.shortcut1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x += identity
        x = F.relu(x)
        x = self.pool(x)
        
        # Residual block 2
        identity = self.shortcut2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.bn6(self.conv6(x))
        x += identity
        x = F.relu(x)
        x = self.pool(x)
        
        # Residual block 3
        identity = self.shortcut3(x)
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.bn8(self.conv8(x))
        x += identity
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten and fully connected
        x = x.view(-1, 512 * 8 * 8)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x
    

class SmallPetCNN(nn.Module):
    def __init__(self):
        super(SmallPetCNN, self).__init__()
        # Magnifying glass 1: Look at basic shapes (3 colors to 32 clues)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # Squinting tool to shrink the picture
        
        # Magnifying glass 2: Look at more details (32 to 64 clues)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Magnifying glass 3: Look closer (64 to 128 clues)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Magnifying glass 4: Look even closer (128 to 256 clues)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Brain part 1: Take 256 * 16 * 16 clues and squish to 256 ideas
        self.fc1 = nn.Linear(256 * 16 * 16, 256)
        self.bn5 = nn.BatchNorm1d(256)  # Helps the brain think smoothly
        
        # Forget some stuff (30% chance) to avoid overthinking
        self.dropout = nn.Dropout(0.3)
        
        # Brain part 2: Final guess (256 ideas to 1: cat or dog)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Zoom 1, squint to 64x64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Zoom 2, squint to 32x32
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Zoom 3, squint to 16x16
        x = F.relu(self.bn4(self.conv4(x)))             # Zoom 4, stay at 16x16
        x = x.view(-1, 256 * 16 * 16)                  # Flatten to 65,536 clues
        x = F.relu(self.bn5(self.fc1(x)))               # Brain squishes to 256 ideas
        x = self.dropout(x)                             # Forget some stuff
        x = torch.sigmoid(self.fc2(x))                  # Final guess (0 to 1)
        return x
    

class SmallerPetCNN(nn.Module):
    def __init__(self):
        super(SmallerPetCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)    # 3 → 32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)   # 32 → 64
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 64 → 128
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) # 128 → 256
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 16 * 16, 128)       # Cut from 256 to 128
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)                   # 128 → 1
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, 256 * 16 * 16)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x