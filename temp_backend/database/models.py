from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float
)
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

# -------------------- ADMIN USERS -------------------- #
class AdminUser(Base):
    __tablename__ = "admin_users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# -------------------- FLAGGED JOB POSTS -------------------- #
class FlaggedPost(Base):
    __tablename__ = "flagged_posts"

    id = Column(Integer, primary_key=True, index=True)
    post_text = Column(Text, nullable=False)
    reason = Column(String(200), nullable=False)
    comments = Column(Text)
    status = Column(String(20), default="Pending")  # Pending / Reviewed / Resolved
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


# -------------------- PREDICTION LOGS -------------------- #
class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    job_text = Column(Text, nullable=False)
    prediction = Column(String(10), nullable=False)  # Fake / Real
    confidence = Column(Float)
    user_ip = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

