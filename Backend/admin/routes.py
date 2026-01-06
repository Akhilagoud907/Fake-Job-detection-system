# admin/routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import AdminUser, FlaggedPost
from auth.utils import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/admin", tags=["Admin"])

# -------------------- ADMIN LOGIN -------------------- #
@router.post("/login")
def admin_login(data: dict, db: Session = Depends(get_db)):
    username = data.get("username")
    password = data.get("password")

    admin = db.query(AdminUser).filter(AdminUser.username == username).first()

    if not admin or not verify_password(password, admin.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    token = create_access_token({"sub": admin.username, "role": "admin"})
    return {"access_token": token, "token_type": "bearer"}

# -------------------- GET FLAGGED POSTS -------------------- #
@router.get("/flagged")
def get_flagged_posts(db: Session = Depends(get_db)):
    posts = db.query(FlaggedPost).all()
    return [
        {
            "id": p.id,
            "post_text": p.post_text,
            "reason": p.reason,
            "comments": p.comments,
            "status": p.status,
            "confidence": p.confidence,
            "created_at": p.created_at.isoformat()
        }
        for p in posts
    ]

# -------------------- EXPORT FLAGGED POSTS -------------------- #
@router.get("/export/flagged")
def export_flagged_posts(db: Session = Depends(get_db)):
    posts = db.query(FlaggedPost).all()
    if not posts:
        raise HTTPException(status_code=404, detail="No flagged posts found")
    
    return {
        "flagged_posts": [
            {
                "id": p.id,
                "post_text": p.post_text,
                "reason": p.reason,
                "comments": p.comments,
                "status": p.status,
                "confidence": p.confidence,
                "created_at": p.created_at.isoformat()
            }
            for p in posts
        ]
    }




