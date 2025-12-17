import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

def preprocess_imdb(data):
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()
    
    df = df.dropna(subset=['review', 'sentiment'])
    df = df.drop_duplicates(subset=['review'])
    
    df['clean_review'] = df['review'].apply(clean_text)
    df['tokens'] = df['clean_review'].apply(word_tokenize)
    df['lemmatized'] = df['tokens'].apply(lemmatize_tokens)
    
    le = LabelEncoder()
    df['sentiment_label'] = le.fit_transform(df['sentiment'])
    
    return df[['review', 'clean_review', 'lemmatized', 'sentiment', 'sentiment_label']]
