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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gather_paths(base_dir):

    dog_dir = os.path.join(base_dir, 'Dog')
    cat_dir = os.path.join(base_dir, 'Cat')

    dog_files = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir) if f.endswith('.jpg')]
    cat_files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith('.jpg')]

    # Filter out corrupt images
    def is_valid_image(file):
        try:
            img = Image.open(file)
            img.verify()  # Verify that it’s a valid image
            return True
        except:
            return False

    dog_files = [f for f in dog_files if is_valid_image(f)]
    cat_files = [f for f in cat_files if is_valid_image(f)]

    return dog_files, cat_files

def create_train_val_split(dog_files, cat_files, val_size=0.2):
    
    dog_train, dog_val = train_test_split(dog_files, test_size=val_size, random_state=42)

    cat_train, cat_val = train_test_split(cat_files, test_size=val_size, random_state=42)

    train_files = dog_train + cat_train
    val_files = dog_val + cat_val

    train_labels = [1] * len(dog_train) + [0] * len(cat_train)  # 1 for dog, 0 for cat
    val_labels = [1] * len(dog_val) + [0] * len(cat_val)  # 1 for dog, 0 for cat

    return train_files, train_labels, val_files, val_labels



# Step 5: Training Loop with Checkpoint
def train_model(model, train_loader, val_loader, epochs=25, name = 'default'):

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)


    best_val_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    print("Entering training loop...", flush=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=True)
        for i, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar with live stats
            batch_acc = (predicted == labels).sum().item() / labels.size(0)
            train_bar.set_postfix({'loss': loss.item(), 'batch_acc': batch_acc})

        train_acc = train_correct / train_total
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=True)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                val_bar.set_postfix({'loss': loss.item(), 'batch_acc': batch_acc})

        val_acc = val_correct / val_total
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss / len(val_loader))

        print(f'Epoch {epoch+1}/{epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{name}_best_model.pt')

    return history

