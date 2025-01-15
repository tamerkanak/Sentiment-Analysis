# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 17:23:22 2024

@author: tamer
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer
import matplotlib.pyplot as plt
import seaborn as sns

# Veri Yükleme ve Temizleme

# Rating'e göre labellanmış data için aktif et (19884 review):
data = pd.read_json("C:/Users/tamer/Desktop/Data Mining/automatic_label.json")

# Manuel şekilde labellanmış data için aktif et (585 review):
#data = pd.read_csv("C:/Users/tamer/Desktop/Data Mining/manuel_label.csv")

df = data.copy()
df = df.drop_duplicates()

# 1-2-4-5 puanlı yorumları filtrelemek için aktif et:
df = df[df["stars"].isin([1, 2, 4, 5])]

# 1 ve 5 puanlı yorumları filtrelemek için aktif et:
#df = df[df["stars"].isin([1, 5])]

# Manuel etiketlenmiş data için aktif et:
#df["label"] = df["sentiment"].apply(lambda x: "Negative" if x == "Negatif" else "Positive")

# Rating'e göre 1 puana Negative, 5 puana Positive atamak için aktif et:
#df["label"] = df["stars"].apply(lambda x: "Negative" if (x == 1) else "Positive")

# Rating'e göre 1-2 puana Negative, 4-5 puana Positive atamak için aktif et:
df["label"] = df["stars"].apply(lambda x: "Negative" if (x == 1 or x == 2) else "Positive")

# Yıldız puanlarına göre yorum sayısını gösteren çubuk grafik
sns.countplot(x="stars", data=df)

# Her bir çubuğun üzerine değerleri yazdırma
for p in plt.gca().patches:
    plt.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.35, p.get_height()+0.3))

plt.title("Manuel Labeling Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.show()

try:
    df['match'] = df.apply(lambda row: 'Match' if (row['stars'] > 3 and row['sentiment'] == 'Pozitif') or (row['stars'] <= 2 and row['sentiment'] == 'Negatif') else 'No Match', axis=1)
    match_counts = df['match'].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(match_counts, labels=match_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Rating and Emotion Matching/No Matching')
    plt.show()
        
    # 1-2 puanlı yorumlar için eşleşme analizi
    df_1_2 = df[df["stars"].isin([1, 2])]
    df_1_2['match'] = df_1_2.apply(lambda row: 'Match' if row['sentiment'] == 'Negatif' else 'No Match', axis=1)
    match_counts_1_2 = df_1_2['match'].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(match_counts_1_2, labels=match_counts_1_2.index, autopct='%1.1f%%', startangle=140)
    plt.title('Match/No Match in 1-2 Star Comments')
    plt.show()

    # 4-5 puanlı yorumlar için eşleşme analizi
    df_4_5 = df[df["stars"].isin([4, 5])]
    df_4_5['match'] = df_4_5.apply(lambda row: 'Match' if row['sentiment'] == 'Pozitif' else 'No Match', axis=1)
    match_counts_4_5 = df_4_5['match'].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(match_counts_4_5, labels=match_counts_4_5.index, autopct='%1.1f%%', startangle=140)
    plt.title('Match/No Match in 4-5 Star Comments')
    plt.show()

except KeyError:
        print("Manually labeled data should be used for match analysis.")

except Exception as e:
    print(f"Bir hata oluştu: {e}")

# Türkçe stopwords ve stemmer
stop_words = set(stopwords.words("turkish"))
stemmer = TurkishStemmer()

# Metin Ön İşleme Fonksiyonu
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-ZğüşöçıİĞÖŞÇ\s]", "", text)  # Özel karakterleri kaldır
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["cleaned_text"] = df["text"].apply(preprocess_text)

# Özellik Çıkarımı (TF-IDF)
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df["cleaned_text"])

# Etiket Kodlama
le = LabelEncoder()
y = le.fit_transform(df["label"])

# Eğitim ve Test Verilerinin Ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modeller ve Eğitim
models = {
    "Logistic Regression": LogisticRegression(),
    "SVC": SVC(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=False),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
    })

# Sonuçları DataFrame olarak göster
results_df = pd.DataFrame(results)
print(results_df)

# Zero-Shot Classification

# bert-base-turkish-cased kullanmak için aktif et:
print("Labeling process started with bert-base-turkish-cased model.")
zero_shot_classifier = pipeline(
    "zero-shot-classification", model="dbmdz/bert-base-turkish-cased", device=-1
)

# bert-base-turkish-sentiment-cased kullanmak için aktif et:
#print("Labeling process started with bert-base-turkish-sentiment-cased model.")    
#zero_shot_classifier = pipeline(
#    "zero-shot-classification", model="savasy/bert-base-turkish-sentiment-cased", device=-1
#)

# xlm-roberta-base kullanmak için aktif et:
#print("Labeling process started with xlm-roberta-base model.")
#zero_shot_classifier = pipeline(
#    "zero-shot-classification", model="xlm-roberta-base", device=-1
#)

candidate_labels = ["Positive", "Negative"]

df["zero_shot_pred"] = df["text"].apply(
    lambda text: zero_shot_classifier(text, candidate_labels)["labels"][0]
)

# Zero-Shot Performans Değerlendirmesi
y_true = df["label"]
y_pred = df["zero_shot_pred"]

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", pos_label="Positive"
)

print(f"Zero-Shot Accuracy: {accuracy:.4f}")
print(f"Zero-Shot Precision: {precision:.4f}")
print(f"Zero-Shot Recall: {recall:.4f}")
print(f"Zero-Shot F1-Score: {f1:.4f}")
