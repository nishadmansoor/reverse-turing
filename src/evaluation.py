from transformers import BertForSequenceClassification, AutoTokenizer
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import string
from sklearn.linear_model import LogisticRegression
import cv2
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

'''device = torch.device("cpu")

os.makedirs("results", exist_ok=True)

#BERT Evaluation
tokenizer = AutoTokenizer.from_pretrained("models/bert_classifier")
bert_model = BertForSequenceClassification.from_pretrained("models/bert_classifier")
bert_model.to(device)

test_df = pd.read_csv("data/processed/test.csv").dropna(subset=["text"])

class BERTDataset(Dataset):
    def __init__(self, text, label, tokenizer, max_length):
        self.texts = text
        self.labels = label
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.texts.iloc[idx]
        token = self.tokenizer(item, max_length = self.max_length, padding = "max_length", truncation = True)
        result = {}
        
        input_id = torch.tensor(token["input_ids"])
        attention_mask = torch.tensor(token["attention_mask"])
        label = torch.tensor(self.labels.iloc[idx])

        result ={
            "input_ids": input_id,
            "attention_mask": attention_mask,
            "label": label
        }
        return result

test_df = pd.read_csv("data/processed/test.csv").dropna(subset=["text"])

test_dataset = BERTDataset(test_df["text"], test_df["label"], tokenizer, 256)
test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False)

def save_model(dataloader, model, path): 
    model.eval()
    all_pred = []
    all_label = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="BERT Predictions")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            pred = torch.argmax(output.logits, dim=1)
            all_pred.extend(pred.cpu().numpy())
            all_label.extend(labels.cpu().numpy())
    df = pd.DataFrame({"true_label": all_label, "pred_label": all_pred})
    df.to_csv(path, index=False)
    print("Predictions saved to ", path)
save_model(test_loader, bert_model, "results/bert_predictions.csv")

#Stylometric Evaluation 
def extract_features(text):
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 0]
    words = text.lower().split()

    avg_sentence_length = np.mean([len(s.split()) for s in sentences])
    sentence_length_variance = np.var([len(s.split()) for s in sentences])
    avg_word_length = np.mean([len(w) for w in words])
    vocab_richness = len(set(words)) / len(words)
    punctuation_density = sum(1 for char in text if char in string.punctuation) / len(text)

    return [avg_sentence_length, sentence_length_variance, avg_word_length, vocab_richness, punctuation_density]


train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train = [extract_features(text) for text in tqdm(train_df["text"], desc="Stylometric Train")]
y_train = train_df["label"].values

X_test = [extract_features(text) for text in tqdm(test_df["text"], desc="Stylometric Test")]
y_test = test_df["label"].values

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

stylo_results = pd.DataFrame({"true_label": y_test, "pred_label": y_pred})
stylo_results.to_csv("results/stylometric_predictions.csv", index=False)
print("Predictions saved to stylometric_predictions.csv")

#OpenCV Evaluation
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

test_df = pd.read_csv("data/processed/test.csv").sample(n=3000, random_state=42)

images = []
labels = []
for idx in range(len(test_df)):
    text = test_df.iloc[idx]["text"]
    label = test_df.iloc[idx]["label"]
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

train_df_cv = pd.read_csv("data/processed/train.csv").sample(n=14000, random_state=42)
print("Generating training images...")
train_images, train_labels = [], []
for idx in range(len(train_df_cv)):
    text = train_df_cv["text"].iloc[idx]
    features = text_to_features(text)
    image = features_to_image(features)
    train_images.append(image)
    train_labels.append(train_df_cv["label"].iloc[idx])

test_df_cv = pd.read_csv("data/processed/test.csv").sample(n=3000, random_state=42)
print("Generating test images...")
test_images, test_labels = [], []
for idx in range(len(test_df_cv)):
    text = test_df_cv["text"].iloc[idx]
    features = text_to_features(text)
    image = features_to_image(features)
    test_images.append(image)
    test_labels.append(test_df_cv["label"].iloc[idx])

train_dataset_cv = HeatMap(train_images, train_labels)
test_dataset_cv = HeatMap(test_images, test_labels)
train_loader_cv = DataLoader(train_dataset_cv, batch_size=32, shuffle=True)
test_loader_cv = DataLoader(test_dataset_cv, batch_size=32, shuffle=False)

# Train CNN
cnn_model = CNN()
cnn_model.to(device)
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
cnn_loss_fn = nn.CrossEntropyLoss()

print("Training CNN...")
for epoch in range(10):
    cnn_model.train()
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader_cv, desc=f"Epoch {epoch+1}/10")):
        images = images.to(device)
        labels = labels.to(device)
        cnn_optimizer.zero_grad()
        output = cnn_model(images)
        loss = cnn_loss_fn(output, labels)
        loss.backward()
        cnn_optimizer.step()
    print(f"  Epoch {epoch+1}/10, Loss: {loss.item():.4f}")

cnn_model.eval()
cv_preds = []
cv_labels = []
with torch.no_grad():
    for images, labels in test_loader_cv:
        images = images.to(device)
        labels = labels.to(device)
        output = cnn_model(images)
        pred = torch.argmax(output, dim=1)
        cv_preds.extend(pred.cpu().numpy())
        cv_labels.extend(labels.cpu().numpy())

pd.DataFrame({"true_label": cv_labels, "prediction": cv_preds}).to_csv("results/opencv_predictions.csv", index=False)
print(f"OpenCV accuracy: {accuracy_score(cv_labels, cv_preds):.4f}")'''


#Load predictions 
bert_df = pd.read_csv("results/bert_predictions.csv")
bert_labels = bert_df["true_label"].values
bert_preds = bert_df["pred_label"].values
print(f"BERT predictions: {len(bert_preds)} samples")

stylometric_df = pd.read_csv("results/stylometric_predictions.csv")
stylometric_labels = stylometric_df["true_label"].values
stylometric_preds = stylometric_df["pred_label"].values
print(f"Stylometric predictions: {len(stylometric_preds)} samples")

opencv_df = pd.read_csv("results/opencv_predictions.csv")
opencv_labels = opencv_df["true_label"].values
opencv_preds = opencv_df["prediction"].values
print(f"OpenCV predictions: {len(opencv_preds)} samples")

#Evaluation Metrics
results = {
    "BERT": (bert_labels, bert_preds),
    "Stylometric": (stylometric_labels, stylometric_preds),
    "OpenCV": (opencv_labels, opencv_preds)
}

for name, (true, pred) in results.items():
    print(f"\n{name} Evaluation:")
    print(f"Accuracy: {accuracy_score(true, pred):.4f}")
    print(f"Precision: {precision_score(true, pred):.4f}")
    print(f"Recall: {recall_score(true, pred):.4f}")
    print(f"F1 Score: {f1_score(true, pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(true, pred))
    print("Classification Report:")
    print(classification_report(true, pred))

cv_test_df = pd.read_csv("data/processed/test.csv").dropna(subset=["text"]).sample(n=3000, random_state=42)
cv_indices = cv_test_df.index

bert_subset = bert_preds[cv_indices]
stylometric_subset = stylometric_preds[cv_indices]
true_subset = bert_labels[cv_indices]

ensemble_preds = []
for b, s, c in zip(bert_subset, stylometric_subset, opencv_preds):
    vote = b + s + c
    ensemble_preds.append(1 if vote >= 2 else 0)
print(f"Ensemble Accuracy: {accuracy_score(true_subset, ensemble_preds):.4f}")