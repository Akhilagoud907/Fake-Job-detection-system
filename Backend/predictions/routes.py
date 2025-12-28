
import os
import csv
import io
import joblib
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database.db import SessionLocal
from database.models import PredictionLog

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

# -------------------- /predict -------------------- #
@router.post("/predict")
def predict(data: PredictRequest, request: Request, db: Session = Depends(get_db)):
    model = request.app.state.model
    vectorizer = request.app.state.vectorizer

    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="ML model not available")
    
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    # Transform input and predict
    X_vec = vectorizer.transform([data.text])
    prediction = model.predict(X_vec)[0]
    confidence = float(model.predict_proba(X_vec).max())

    # Save prediction to DB
    new_pred = PredictionLog(
        job_text=data.text,
        prediction=str(prediction),
        confidence=confidence
    )
    db.add(new_pred)
    db.commit()
    db.refresh(new_pred)

    return {
        "prediction": str(prediction),
        "confidence": confidence
    }

# -------------------- /export -------------------- #
@router.get("/export")
def export_predictions(db: Session = Depends(get_db)):
    predictions = db.query(PredictionLog).all()
    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions found")

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Job Text", "Prediction", "Confidence", "Created At"])
    
    for p in predictions:
        writer.writerow([p.id, p.job_text, p.prediction, p.confidence, p.created_at.isoformat()])

    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions.csv"}
    )







