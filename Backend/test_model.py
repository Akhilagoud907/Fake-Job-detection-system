import joblib

MODEL_PATH = "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print(model)  # Shows model type and parameters
except EOFError:
    print("❌ Model file is empty or corrupted!")
except Exception as e:
    print("❌ Error loading model:", e)
