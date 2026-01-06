# predictions/routes.py
import csv
import io
import time
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import PredictionLog, FlaggedPost

# -------------------- ROUTER SETUP -------------------- #
router = APIRouter(prefix="/predictions", tags=["Predictions"])

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

    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="ML model not available")

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
    db.refresh(new_prediction)

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
    post_text = data.get("text")
    reason = data.get("reason", "")
    comments = data.get("comments", "")

    if not post_text:
        raise HTTPException(status_code=400, detail="Post text is required")

    new_post = FlaggedPost(
        post_text=post_text,
        reason=reason,
        comments=comments,
        status="Flagged",   # default status
        confidence=None     # optional
    )

    db.add(new_post)
    db.commit()
    db.refresh(new_post)

    return {
        "status": "success",
        "id": new_post.id,
        "post_text": new_post.post_text,
        "reason": new_post.reason
    }

# -------------------- EXPORT PREDICTIONS -------------------- #
@router.get("/export")
def export_predictions(db: Session = Depends(get_db)):
    predictions = db.query(PredictionLog).all()
    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions found")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Job Text", "Prediction", "Confidence", "Created At"])

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
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )

# -------------------- EXPORT FLAGGED POSTS -------------------- #
@router.get("/export/flagged")
def export_flagged_posts(db: Session = Depends(get_db)):
    posts = db.query(FlaggedPost).all()
    if not posts:
        raise HTTPException(status_code=404, detail="No flagged posts found")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Job Text", "Reason", "Comments", "Status", "Confidence", "Created At"])

    for p in posts:
        writer.writerow([
            p.id,
            p.post_text,
            p.reason,
            p.comments,
            p.status,
            p.confidence if p.confidence is not None else "",
            p.created_at.isoformat()
        ])

    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=flagged_posts.csv"}
    )

# -------------------- PREDICTIONS SUMMARY -------------------- #
@router.get("/summary")
def get_predictions_summary(db: Session = Depends(get_db)):
    """
    Returns aggregated data for dashboard:
    - total predictions
    - count of Fake vs Real
    - percentage of Fake vs Real
    """
    total = db.query(PredictionLog).count()
    fake = db.query(PredictionLog).filter(PredictionLog.prediction == "Fake").count()
    real = db.query(PredictionLog).filter(PredictionLog.prediction == "Real").count()
    
    fake_percent = round((fake / total * 100), 2) if total else 0
    real_percent = round((real / total * 100), 2) if total else 0
    
    return {
        "total": total,
        "fake": fake,
        "real": real,
        "fake_percent": fake_percent,
        "real_percent": real_percent
    }










