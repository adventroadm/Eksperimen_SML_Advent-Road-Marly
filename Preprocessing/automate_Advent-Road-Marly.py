import pandas as pd
import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import numpy as np

import os
import nltk

nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download resource
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

# Inisialisasi Lemmatizer dan Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Membersihkan teks: lowercase, hapus HTML, tanda baca, angka, spasi berlebih."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)  # hapus HTML tags
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
    text = re.sub(r"\d+", "", text)  # hapus angka
    text = re.sub(r"\s+", " ", text).strip()  # hapus spasi berlebih
    return text

def lemmatize_tokens(tokens):
    """Lemmatization dan hapus stopwords."""
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

def compute_embedding(tokens_list, model):
    """Hitung embedding rata-rata per dokumen (review)."""
    vectors = []
    for tokens in tokens_list:
        valid_tokens = [t for t in tokens if t in model.wv]
        if valid_tokens:
            vectors.append(np.mean(model.wv[valid_tokens], axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return vectors

def preprocess_imdb(data, embedding_size=100, window=5, min_count=2, sg=1):
    # Baca CSV jika input berupa path
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()
    
    # Hapus missing values & duplikasi
    df = df.dropna(subset=['review', 'sentiment'])
    df = df.drop_duplicates(subset=['review'])
    
    # Cleaning teks
    df['clean_review'] = df['review'].apply(clean_text)
    
    # Tokenisasi
    df['tokens'] = df['clean_review'].apply(word_tokenize)
    
    # Lemmatization + hapus stopwords
    df['lemmatized'] = df['tokens'].apply(lemmatize_tokens)
    
    # Encode label sentiment
    le = LabelEncoder()
    df['sentiment_label'] = le.fit_transform(df['sentiment'])
    
    # Latih Word2Vec di seluruh lemmatized tokens
    sentences = df['lemmatized'].tolist()
    w2v_model = Word2Vec(
        sentences,
        vector_size=embedding_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=sg
    )
    
    # Hitung embedding rata-rata per review
    df['embedding'] = compute_embedding(df['lemmatized'], w2v_model)
    
    return df[['review', 'clean_review', 'lemmatized', 'sentiment', 'sentiment_label', 'embedding']], w2v_model

if __name__ == "__main__":
    # Input CSV
    input_csv = "IMDB_raw/IMDB_raw.csv"
    
    # Folder output
    output_folder = "Preprocessing"
    os.makedirs(output_folder, exist_ok=True)
    
    # Jalankan preprocessing
    df_processed = preprocess_imdb(input_csv)
    
    # Simpan CSV hasil preprocess
    csv_path = os.path.join(output_folder, "IMDB_preprocessing.csv")
    df_processed.to_csv(csv_path, index=False)
    print(f"CSV hasil preprocess tersimpan di: {csv_path}")
