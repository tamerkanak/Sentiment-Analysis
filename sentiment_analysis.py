# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 17:23:22 2024

@author: tamer
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from transformers import pipeline
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# DataFrame'i yükle
data = pd.read_csv('C:/Users/tamer/Desktop/Data Mining/review_analysis_results.csv')

data = pd.read_json('C:/Users/tamer/Desktop/Data Mining/review_balanced.json')
df = pd.DataFrame(data)

# Sadece 1 ve 5 puanları filtrele
df = df[df['sentiment'].isin(["Negatif", "Pozitif"])].copy()

# Denetimli Duygu Analizi
# 1. Veri Ön İşleme ve Etiketleme
df = df[(df['stars'] == 1) | (df['stars'] == 5) | (df['stars'] == 4) | (df['stars'] == 2)]
df['label'] = df['sentiment'].apply(lambda x: 'Negative' if x == "Negatif" else 'Positive')

# Türkçe stemmer'ı başlat
stemmer = TurkishStemmer()

# Stopwords ve tokenizer'ı nltk'den alıyoruz
stop_words = set(stopwords.words('turkish'))

# Preprocessing fonksiyonu
def preprocess_text(text):
    # Küçük harfe çevir ve özel karakterleri kaldır
    text = text.lower()
    text = re.sub(r'[^a-zA-ZğüşöçıİĞÖŞÇ\s]', '', text)  # Özel karakterleri kaldır
    
    # Tokenization (kelimelere ayırma)
    tokens = word_tokenize(text)
    
    # Stopwords kaldırma ve stemming işlemi
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Preprocessing işlemi
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Feature Extraction (TF-IDF Vektörizasyonu)
vectorizer = TfidfVectorizer(max_features=10000)  # 5000 özellik kullanacağız
X = vectorizer.fit_transform(df['cleaned_text'])  # Veriyi vektörize et

# Label Encoding işlemi
le = LabelEncoder()
y = le.fit_transform(df['label'])  # 'Negative' -> 0, 'Positive' -> 1

# Eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Denemek istediğiniz algoritmalar
models = [
    ('Logistic Regression', LogisticRegression()),
    ('SVC', SVC()),
    ('Random Forest', RandomForestClassifier()),
    ('AdaBoost', AdaBoostClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('XGBoost', XGBClassifier()),
    ('LightGBM', LGBMClassifier()),
    ('CatBoost', CatBoostClassifier(verbose=False))
]

# Sonuçları saklamak için bir DataFrame oluşturma
results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

# Her bir model için eğitim, test ve değerlendirme
for name, model in models:
    # Model eğitimi
    model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = model.predict(X_test)
    
    # Metrikler
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Sonuçları DataFrame'e ekleme
    new_row = pd.DataFrame({
        'Model': [name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1]
    })
    
    results = pd.concat([results, new_row], ignore_index=True)
    
y = df['label']  # 'sentiment' yerine 'label' kullanacağız

# Zero-Shot Classification pipeline'ını başlat
zero_shot_classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased", device=0)

# Pozitif ve Negatif etiketlerini belirleyelim
candidate_labels = ['Positive', 'Negative']

# Zero-shot sentiment analysis fonksiyonu
def zero_shot_analysis(text):
    result = zero_shot_classifier(text, candidate_labels)
    return result['labels'][0]  # En yüksek olasılıkla tahmin edilen etiketi döndür

# Zero-Shot Analizi Uygulama
df['zero_shot_pred'] = df['cleaned_text'].apply(zero_shot_analysis)

# Model Performansını Değerlendirme
# Gerçek etiketler ve model tahminleri
y_true = df['label']
y_pred = df['zero_shot_pred']

# Model Performansını Değerlendirme
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label='Positive')

# Sonuçları Yazdırma
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')