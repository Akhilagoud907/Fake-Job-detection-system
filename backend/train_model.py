# train_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------------------
# 1. Sample Dataset
# ---------------------------
# You can replace this with your real dataset later
data = {
    "text": [
        "Software engineer needed at a top tech company",
        "Earn $5000 weekly from home, no experience required",
        "Looking for data analyst with 2 years experience",
        "Work from home and make money fast",
        "Senior Python developer required",
        "Get rich quick by clicking this link"
    ],
    "label": [0, 1, 0, 1, 0, 1]  # 0 = Real, 1 = Fake
}

df = pd.DataFrame(data)

# ---------------------------
# 2. Split dataset
# ---------------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 3. TF-IDF Vectorizer
# ---------------------------
vectorizer = TfidfVectorizer(max_features=500)
X_train_vec = vectorizer.fit_transform(X_train)

# ---------------------------
# 4. Train Model
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# ---------------------------
# 5. Save Model & Vectorizer
# ---------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
