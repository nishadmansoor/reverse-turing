import numpy as np
import pandas as pd
import string
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

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
val_df = pd.read_csv("data/processed/val.csv")

X_train = [extract_features(text) for text in train_df["text"]]
y_train = train_df["label"].values

X_val = [extract_features(text) for text in val_df["text"]]
y_val = val_df["label"].values

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"Validation Accuracy: {accuracy:.2f}")