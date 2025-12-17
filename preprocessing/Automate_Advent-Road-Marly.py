import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import mlflow
import os

# Pastikan resource nltk sudah di-download
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Inisialisasi Lemmatizer dan Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ------------------------------
# SET TRACKING URI MLflow
# ------------------------------
# Pilih salah satu:
mlflow.set_tracking_uri("http://localhost:5000")

def clean_text(text):
    """Membersihkan teks: lowercase, hapus HTML, tanda baca, angka, spasi berlebih."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)  # hapus HTML tags
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
    text = re.sub(r"\d+", "", text)  # hapus angka
    text = re.sub(r"\s+", " ", text).strip()  # hapus spasi berlebih
    return text

def lemmatize_tokens(tokens):
    """Lemmatization dan hapus stopwords."""
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

def preprocess_imdb(data, save_path=None):
    with mlflow.start_run():  # mulai MLflow run
        # Jika input berupa path CSV, baca dulu
        if isinstance(data, str):
            df = pd.read_csv(data)
            mlflow.log_param("input_path", data)
        else:
            df = data.copy()
            mlflow.log_param("input_type", "DataFrame")

        initial_count = len(df)
        mlflow.log_param("initial_rows", initial_count)

        # Hapus missing values & duplikasi
        df = df.dropna(subset=['review', 'sentiment'])
        df = df.drop_duplicates(subset=['review'])
        cleaned_count = len(df)
        mlflow.log_param("cleaned_rows", cleaned_count)

        # Cleaning teks
        df['clean_review'] = df['review'].apply(clean_text)

        # Tokenisasi
        df['tokens'] = df['clean_review'].apply(word_tokenize)

        # Lemmatization + hapus stopwords
        df['lemmatized'] = df['tokens'].apply(lemmatize_tokens)

        # Encode label sentiment
        le = LabelEncoder()
        df['sentiment_label'] = le.fit_transform(df['sentiment'])
        mlflow.log_param("label_mapping", dict(zip(le.classes_, le.transform(le.classes_))))

        # Simpan hasil preprocessing sebagai artifact CSV jika path diberikan
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            mlflow.log_artifact(save_path, artifact_path="preprocessed_data")

        return df[['review', 'clean_review', 'tokens', 'lemmatized', 'sentiment', 'sentiment_label']]

# ------------------------------
# Contoh pemanggilan
# ------------------------------
df_preprocessed = preprocess_imdb(
    "IMDB_raw.csv", 
    save_path="preprocessing/IMDB_preprocessing.csv"
)