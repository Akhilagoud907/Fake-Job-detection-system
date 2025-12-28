# ==============================================
# Milestone1 - OneDrive-safe + EDA + WordClouds (Live + Saved PNG)
# ==============================================

# Imports & setup
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle
from scipy import sparse
import joblib

# Display options
pd.set_option('display.max_colwidth', 200)
sns.set(style='whitegrid')

print('Libraries loaded. Current working directory:', os.getcwd())

# ==============================================
# Ensure C:\Temp folder exists
# ==============================================
temp_dir = r'C:\Temp'
os.makedirs(temp_dir, exist_ok=True)

# ==============================================
# CSV loading (OneDrive-safe)
# ==============================================
source_csv = r'C:\Users\akhil\OneDrive\Desktop\Fake_job\Milestone1\fake_job_postings.csv'
csv_path = os.path.join(temp_dir, 'fake_job_postings.csv')

if not os.path.exists(csv_path):
    shutil.copy(source_csv, csv_path)
    print(f"CSV copied to safe local folder: {csv_path}")

# Load CSV
df = pd.read_csv(csv_path)
print('Loaded dataset shape:', df.shape)

# ==============================================
# Detect target column
# ==============================================
possible_targets = ['fraudulent', 'fraud', 'is_fake', 'label', 'target']
target_col = None
for c in df.columns:
    if c.lower() in possible_targets:
        target_col = c
        break
if target_col is None:
    for c in df.columns:
        if df[c].dropna().isin([0,1]).all():
            target_col = c
            break
print('Target column detected:', target_col)

# ==============================================
# NLTK setup
# ==============================================
nltk_data_needed = ['stopwords', 'wordnet', 'omw-1.4']
for resource in nltk_data_needed:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()  # Safe tokenizer

# ==============================================
# Text cleaning
# ==============================================
def clean_text(text, remove_stopwords=True, lemmatize=True):
    if pd.isnull(text):
        return ''
    text = BeautifulSoup(str(text), 'html.parser').get_text(separator=' ')
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|\S+@\S+', ' ', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = tokenizer.tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# ==============================================
# Apply cleaning to text columns
# ==============================================
text_cols = ['title', 'location', 'department', 'company_profile', 'description', 
             'requirements', 'benefits', 'employment_type', 'required_experience']
existing_text_cols = [c for c in text_cols if c in df.columns]

for c in existing_text_cols:
    df[c+'_clean'] = df[c].astype(str).apply(clean_text)

combine_cols = [c+'_clean' for c in existing_text_cols]
df['text'] = df[combine_cols].agg(' '.join, axis=1)

# ==============================================
# Handle missing values
# ==============================================
for c in df.columns:
    if df[c].dtype == 'object' and not c.endswith('_clean') and c != 'text':
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else '')
    elif np.issubdtype(df[c].dtype, np.number) and df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())

# ==============================================
# EDA - Target distribution (Show + Save)
# ==============================================
if target_col:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[target_col])
    plt.title('Target distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, 'target_distribution.png'))
    plt.show()  # <-- display in terminal
    print("Target distribution plot saved and displayed.")

# ==============================================
# WordClouds for fake vs real (Show + Save)
# ==============================================
def display_wordcloud(texts, title, filename):
    wc = WordCloud(width=800, height=400, collocations=False, background_color='white').generate(' '.join(texts))
    plt.figure(figsize=(12,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, filename))
    plt.show()  # <-- display in terminal

if 'text' in df.columns and target_col:
    for val in df[target_col].unique():
        subset = df[df[target_col]==val]['text'].dropna().tolist()
        display_wordcloud(subset, f'WordCloud for {target_col}={val}', f'wordcloud_{target_col}_{val}.png')

# ==============================================
# Top words per class - save as CSV
# ==============================================
def top_n_words(texts, n=25):
    cnt = Counter()
    for t in texts:
        cnt.update(t.split())
    return cnt.most_common(n)

top_words_dict = {}
if 'text' in df.columns and target_col:
    for val in df[target_col].unique():
        texts = df[df[target_col]==val]['text'].dropna().tolist()
        top_words_dict[val] = top_n_words(texts, 25)

# Save top words per class
top_words_df = pd.DataFrame({
    val: dict(top_words_dict[val]) for val in top_words_dict
})
top_words_df.to_csv(os.path.join(temp_dir, 'top_words_per_class.csv'))
print("Top words per class saved as CSV.")

# ==============================================
# Feature extraction: TF-IDF
# ==============================================
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
X_tfidf = tfidf_vectorizer.fit_transform(df['text'].fillna(''))

# Labels
y = df[target_col].values

# ==============================================
# Train/test split
# ==============================================
X_train_tfidf, X_test_tfidf, y_train, y_test, X_train_raw, X_test_raw = train_test_split(
    X_tfidf, y, df['text'].values, test_size=0.2, random_state=42, stratify=y
)

# ==============================================
# Save artifacts for Milestone 2
# ==============================================
sparse.save_npz(os.path.join(temp_dir, 'X_train_tfidf.npz'), X_train_tfidf)
sparse.save_npz(os.path.join(temp_dir, 'X_test_tfidf.npz'), X_test_tfidf)
joblib.dump(X_train_raw, os.path.join(temp_dir, 'X_train_raw.pkl'))
joblib.dump(X_test_raw, os.path.join(temp_dir, 'X_test_raw.pkl'))
joblib.dump(y_train, os.path.join(temp_dir, 'y_train.pkl'))
joblib.dump(y_test, os.path.join(temp_dir, 'y_test.pkl'))
with open(os.path.join(temp_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
df.to_csv(os.path.join(temp_dir, 'fake_job_postings_clean.csv'), index=False)

print("\nAll Milestone 1 artifacts saved to C:\\Temp, including WordClouds, top words, and live plot display.")












