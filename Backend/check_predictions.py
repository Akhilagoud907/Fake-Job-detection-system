from database.db import SessionLocal
from database.models import PredictionLog

# Create a database session
db = SessionLocal()

# Total predictions
total = db.query(PredictionLog).count()
fake = db.query(PredictionLog).filter(PredictionLog.prediction == "Fake").count()
real = db.query(PredictionLog).filter(PredictionLog.prediction == "Real").count()

# Print results
print("Total predictions:", total)
print("Fake:", fake)
print("Real:", real)

# Optional: List the latest 5 predictions
print("\nLatest 5 predictions:")
for p in db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(5):
    print(f"ID: {p.id}, Text: {p.job_text}, Prediction: {p.prediction}, Confidence: {p.confidence}")
