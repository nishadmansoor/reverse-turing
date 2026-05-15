import cv2
import numpy as np
import pandas as pd
import string
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def text_to_features(text):
    sentences = text.split(".")
    features = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > 0:
            word_count = len(words)
            avg_word_length = sum(len(word) for word in words) / len(words)
            puncutation = sum(1 for char in sentence if char in string.punctuation)
            features.append([len(sentence), word_count, avg_word_length, puncutation])
    return features


def features_to_image(features):
    grid = np.array(features, dtype=np.float32)
    grid = cv2.normalize(grid, None, 0, 255, cv2.NORM_MINMAX)
    grid = grid.astype(np.uint8)
    image = cv2.resize(grid, (224, 224))
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return heatmap

train_df = pd.read_csv("data/processed/train.csv").sample(n=14000, random_state=42)

images = []
labels = []
for idx in range(len(train_df)):
    text = train_df.iloc[idx]["text"]
    label = train_df.iloc[idx]["label"]
    features = text_to_features(text)
    image = features_to_image(features)
    images.append(image)
    labels.append(label)

print(f"Generated {len(images)} images")
print(f"Image shape: {images[0].shape}")
print(f"Labels: {sum(labels)} AI, {len(labels) - sum(labels)} human")

class HeatMap(Dataset):
    def __init__(self, images, labels):
        # store images and labels
        self.images = images
        self.labels = labels

    def __len__(self):
        # return count
        return len(self.images)

    def __getitem__(self, idx):
        # get image and label
        image = self.images[idx]
        label = self.labels[idx]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(label)
        return image, label

test_df = pd.read_csv("data/processed/test.csv").sample(n=3000, random_state=42)
val_df = pd.read_csv("data/processed/val.csv").sample(n=3000, random_state=42)

train_dataset = HeatMap(images, labels)

val_images, val_labels = [], []
for idx in range(len(val_df)):
    text = val_df["text"].iloc[idx]
    features = text_to_features(text)
    image = features_to_image(features)
    val_images.append(image)
    val_labels.append(val_df["label"].iloc[idx])

val_dataset = HeatMap(val_images, val_labels)

test_images, test_labels = [], []
for idx in range(len(test_df)):
    text = test_df["text"].iloc[idx]
    features = text_to_features(text)
    image = features_to_image(features)
    test_images.append(image)
    test_labels.append(test_df["label"].iloc[idx])

test_dataset = HeatMap(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

device = torch.device("cpu")
model.to(device)

def train(dataloader, model, optimizer, loss_fn):
    model.train()
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = loss_fn(output, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}")
    train(train_loader, model, optimizer, loss_fn)

def validate(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            pred = torch.argmax(output, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Validation Accuracy: {100 * accuracy:.2f}")

validate(val_loader, model)