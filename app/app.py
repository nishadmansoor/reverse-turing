from flask import Flask, request, render_template
import torch
import numpy as np
import string
from transformers import BertForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__)

print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained("models/bert_classifier")
bert_model = BertForSequenceClassification.from_pretrained("models/bert_classifier")
bert_model.eval()

def extract_features(text):
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 0]
    words = text.lower().split()
    if len(sentences) == 0 or len(words) == 0:
        return [0, 0, 0, 0, 0]
    avg_sentence_length = np.mean([len(s.split()) for s in sentences])
    sentence_length_variance = np.var([len(s.split()) for s in sentences])
    avg_word_length = np.mean([len(w) for w in words])
    vocab_richness = len(set(words)) / len(words)
    punctuation_density = sum(1 for char in text if char in string.punctuation) / len(text)
    return [avg_sentence_length, sentence_length_variance, avg_word_length, vocab_richness, punctuation_density]

train_df = pd.read_csv("data/processed/train.csv").dropna(subset=["text"])
X_train = [extract_features(text) for text in train_df["text"]]
y_train = train_df["label"].values
stylo_model = LogisticRegression(max_iter=1000)
stylo_model.fit(X_train, y_train)
X_train_arr = np.array(X_train)
human_avg = X_train_arr[y_train == 0].mean(axis=0)
ai_avg = X_train_arr[y_train == 1].mean(axis=0)
print("Models loaded")

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods = ["POST"])
def predict():
    text = request.form["text"]
    #BERT
    tokens = tokenizer(text, max_length=256, padding = "max_length", truncation = True, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens)
        probs = torch.softmax(output.logits, dim=1)
        bert_pred = torch.argmax(probs, dim=1).item()
        bert_confidence = round(probs[0][bert_pred].item() * 100, 1)
        bert_ai_prob = round(probs[0][1].item() * 100, 1)
    #Stylometric
    features = extract_features(text) 
    features_arr = np.array(extract_features(text)).reshape(1,-1)
    stylo_pred = stylo_model.predict(features_arr)[0]
    stylo_proba = stylo_model.predict_proba(features_arr)[0]
    stylo_confidence = round(stylo_proba[stylo_pred] * 100, 1)
    stylo_ai_prob = round(stylo_proba[1] * 100, 1)

    #Feature Comparison
    feature_comparison = []
    names = ["Avg Sentence Length", "Sentence Length Variance", "Avg Word Length", "Vocabulary Richness", "Punctuation Density"]
    for i, name in enumerate(names):
        feature_comparison.append({
            "name": name,
            "user_val": round(features[i], 2),
            "human_avg": round(human_avg[i], 2),
            "ai_avg": round(ai_avg[i], 2),
        })

    #Results 
    results = {
            "text": text[:300],
            "bert": "AI" if bert_pred == 1 else "Human",
            "bert_confidence": bert_confidence,
            "bert_ai_prob": bert_ai_prob,
            "stylometric": "AI" if stylo_pred == 1 else "Human",
            "stylo_confidence": stylo_confidence,
            "stylo_ai_prob": stylo_ai_prob,
            "features": feature_comparison,
        }
    return render_template("results.html", results=results)
if __name__ == "__main__":
    app.run(debug=True)