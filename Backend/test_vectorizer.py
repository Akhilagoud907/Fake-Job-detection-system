import joblib

VECTORIZER_PATH = "vectorizer.pkl"

try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Vectorizer loaded successfully!")
    print(vectorizer)
except EOFError:
    print("❌ Vectorizer file is empty or corrupted!")
except Exception as e:
    print("❌ Error loading vectorizer:", e)
