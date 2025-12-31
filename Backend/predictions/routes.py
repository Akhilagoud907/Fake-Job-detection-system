import csv
import io
import time
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.db import SessionLocal
from database.models import PredictionLog

# -------------------- ROUTER SETUP -------------------- #
router = APIRouter(prefix="/predictions", tags=["Predictions"])

# -------------------- DB DEPENDENCY -------------------- #
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------- REQUEST MODEL -------------------- #
class PredictRequest(BaseModel):
    text: str

# -------------------- PREDICT ENDPOINT -------------------- #
@router.post("/predict")
def predict(
    data: PredictRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    start_time = time.time()

    model = request.app.state.model
    vectorizer = request.app.state.vectorizer

    # Model availability check
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="ML model not available")

    # Input validation
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    # Vectorize input
    X_vec = vectorizer.transform([data.text])

    # Prediction
    pred_value = model.predict(X_vec)[0]
    confidence = float(model.predict_proba(X_vec).max())

    prediction_label = "Fake" if pred_value == 1 else "Real"

    processing_time = round(time.time() - start_time, 3)

    # Save prediction to database
    new_prediction = PredictionLog(
        job_text=data.text,
        prediction=prediction_label,
        confidence=confidence
    )
    db.add(new_prediction)
    db.commit()

    # API Response
    return {
        "prediction": prediction_label,
        "confidence": round(confidence * 100, 2),
        "processing_time": processing_time
    }

# -------------------- FLAG POST ENDPOINT -------------------- #
@router.post("/flag")
async def flag_post(data: dict, db: Session = Depends(get_db)):
    """
    Save flagged job post to the database
    """
    print("Flagged post received:", data)
    # Optional: Save to DB table here
    return {"status": "success"}

# -------------------- EXPORT PREDICTIONS -------------------- #
@router.get("/export")
def export_predictions(db: Session = Depends(get_db)):
    predictions = db.query(PredictionLog).all()

    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions found")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID",
        "Job Text",
        "Prediction",
        "Confidence",
        "Created At"
    ])

    for p in predictions:
        writer.writerow([
            p.id,
            p.job_text,
            p.prediction,
            round(p.confidence * 100, 2),
            p.created_at.isoformat()
        ])

    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=predictions.csv"
        }
    )







