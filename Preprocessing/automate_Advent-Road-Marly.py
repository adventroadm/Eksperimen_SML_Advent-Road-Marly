import pandas as pd
import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

# Setup NLTK
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

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

def preprocess_imdb_tfidf(data, max_features=5000):
    """Preprocessing dataset IMDB dan membuat representasi TF-IDF."""
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
    
    # Tokenisasi + Lemmatization
    df['tokens'] = df['clean_review'].apply(word_tokenize)
    df['lemmatized'] = df['tokens'].apply(lemmatize_tokens)
    
    # Gabungkan tokens menjadi string untuk TF-IDF
    df['lemmatized_text'] = df['lemmatized'].apply(lambda x: " ".join(x))
    
    # Encode label sentiment
    le = LabelEncoder()
    df['sentiment_label'] = le.fit_transform(df['sentiment'])
    
    # TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['lemmatized_text'])
    
    return df[['review', 'clean_review', 'lemmatized', 'sentiment', 'sentiment_label']], tfidf_matrix, tfidf_vectorizer

if __name__ == "__main__":
    # Input CSV
    input_csv = "IMDB_raw/IMDB_raw.csv"
    
    # Folder output
    output_folder = "Preprocessing"
    os.makedirs(output_folder, exist_ok=True)
    
    # Jalankan preprocessing TF-IDF
    df_processed, tfidf_matrix, tfidf_vectorizer = preprocess_imdb_tfidf(input_csv)
    
    # Simpan CSV hasil preprocess
    csv_path = os.path.join(output_folder, "IMDB_preprocessing.csv")
    df_processed.to_csv(csv_path, index=False)
    print(f"CSV hasil preprocess tersimpan di: {csv_path}")
    
    # TF-IDF matrix bisa disimpan sebagai npz jika mau
    tfidf_path = os.path.join(output_folder, "IMDB_tfidf.npz")
    from scipy.sparse import save_npz
    save_npz(tfidf_path, tfidf_matrix)
    print(f"TF-IDF matrix tersimpan di: {tfidf_path}")
