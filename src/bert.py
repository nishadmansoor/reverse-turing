from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_df = pd.read_csv("data/processed/train.csv").dropna(subset=["text"])
#sample = train_df["text"].iloc[0]
#token = tokenizer(sample, max_length = 512, padding = "max_length", truncation = True)
#print(token.keys())

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
val_df = pd.read_csv("data/processed/val.csv").dropna(subset=["text"])

train_dataset = BERTDataset(train_df["text"], train_df["label"], tokenizer, 256)
test_dataset = BERTDataset(test_df["text"], test_df["label"], tokenizer, 256)
val_dataset = BERTDataset(val_df["text"], val_df["label"], tokenizer, 256)


train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False)
val_loader =  DataLoader(val_dataset, batch_size = 8, shuffle = False)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

def train(dataloader, model, optimizer):
    model.train()

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        if torch.isnan(output.loss):
            print(f"Nan loss at batch {batch_idx}")
            continue

        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {output.loss.item(): 4f}")


epochs = 3
for t in range(epochs):
    print(f"Epoch {t+1}")
    train(train_loader, model, optimizer)
print("Done")

model.save_pretrained("models/bert_classifier")
tokenizer.save_pretrained("models/bert_classifier")


def validation(dataloader, model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            pred = torch.argmax(output.logits, dim=1)
            correct += (pred==labels).sum().item()
            total += labels.size(0)
    accuracy = correct/total
    print(f"Validation Accuracy: {100*accuracy:.2f}")

validation(val_loader, model)

def save_model(dataloader, model, path): 
    model.eval()
    all_pred = []
    all_label = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
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
    