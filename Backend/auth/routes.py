from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database.db import SessionLocal
from database.models import AdminUser, FlaggedPost
from auth.utils import hash_password, verify_password, create_access_token

router = APIRouter(tags=["Admin"])

# -------------------- DB DEPENDENCY -------------------- #
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------- ADMIN LOGIN -------------------- #
@router.post("/auth/login")
def admin_login(data: dict, db: Session = Depends(get_db)):
    username = data.get("username")
    password = data.get("password")

    admin = db.query(AdminUser).filter(AdminUser.username == username).first()

    if not admin or not verify_password(password, admin.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    token = create_access_token({
        "sub": admin.username,
        "role": "admin"
    })

    return {
        "access_token": token,
        "token_type": "bearer"
    }

# -------------------- GET FLAGGED POSTS -------------------- #
@router.get("/admin/flagged")
def get_flagged_posts(db: Session = Depends(get_db)):
    flagged_posts = db.query(FlaggedPost).all()
    result = [
        {
            "id": f.id,
            "post_text": f.post_text,
            "reason": f.reason,
            "comments": f.comments,
            "status": f.status,
            "confidence": f.confidence,
            "created_at": f.created_at.isoformat()
        }
        for f in flagged_posts
    ]
    return {"flagged_posts": result}

# -------------------- EXPORT FLAGGED POSTS -------------------- #
@router.get("/admin/export/flagged")
def export_flagged_posts(db: Session = Depends(get_db)):
    flagged_posts = db.query(FlaggedPost).all()
    if not flagged_posts:
        raise HTTPException(status_code=404, detail="No flagged posts found")

    result = [
        {
            "id": f.id,
            "post_text": f.post_text,
            "reason": f.reason,
            "comments": f.comments,
            "status": f.status,
            "confidence": f.confidence,
            "created_at": f.created_at.isoformat()
        }
        for f in flagged_posts
    ]
    return {"flagged_posts": result}

# -------------------- ADMIN STATS -------------------- #
@router.get("/admin/stats")
def get_admin_stats(db: Session = Depends(get_db)):
    total_flagged = db.query(FlaggedPost).count()
    pending = db.query(FlaggedPost).filter(FlaggedPost.status == "Pending").count()
    reviewed = db.query(FlaggedPost).filter(FlaggedPost.status == "Reviewed").count()
    resolved = db.query(FlaggedPost).filter(FlaggedPost.status == "Resolved").count()

    return {
        "total_flagged": total_flagged,
        "pending": pending,
        "reviewed": reviewed,
        "resolved": resolved
    }




