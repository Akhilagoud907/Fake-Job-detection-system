# db.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# ----------------------------
# 1️⃣ Database Configuration
# ----------------------------
DATABASE_URL = "sqlite:///database.db"  # Change path if needed

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# ----------------------------
# 2️⃣ Prediction Model
# ----------------------------
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(String)
    company_profile = Column(String, nullable=True)
    location = Column(String, nullable=True)
    employment_type = Column(String, nullable=True)
    prediction = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# ----------------------------
# 3️⃣ Create Tables
# ----------------------------
Base.metadata.create_all(bind=engine)

# ----------------------------
# 4️⃣ Dependency for FastAPI
# ----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------------------
# 5️⃣ Helper Functions
# ----------------------------

def save_prediction(db: Session, job_dict, prediction_result):
    """
    Save a prediction to the database.
    """
    pred = Prediction(
        title=job_dict.get("title"),
        description=job_dict.get("description"),
        company_profile=job_dict.get("company_profile"),
        location=job_dict.get("location"),
        employment_type=job_dict.get("employment_type"),
        prediction=prediction_result,
        created_at=datetime.utcnow()
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred

def get_all_predictions(db: Session):
    """
    Fetch all predictions from the database (latest first).
    """
    return db.query(Prediction).order_by(Prediction.created_at.desc()).all()

