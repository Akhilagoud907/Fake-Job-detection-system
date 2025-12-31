import os
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from database.db import engine, SessionLocal
from database.models import Base, AdminUser
from auth.utils import hash_password

from auth.routes import router as auth_router
from admin.routes import router as admin_router
from predictions.routes import router as prediction_router

# -------------------- APP SETUP -------------------- #
app = FastAPI(title="Fake Job Detection System")

# -------------------- CORS -------------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or limit to frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- DATABASE INIT -------------------- #
Base.metadata.create_all(bind=engine)

# -------------------- SEED DEFAULT ADMIN -------------------- #
def create_default_admin():
    db = SessionLocal()
    try:
        admin_exists = db.query(AdminUser).first()
        if not admin_exists:
            admin = AdminUser(
                username="admin",
                password_hash=hash_password("admin123")
            )
            db.add(admin)
            db.commit()
            print("✅ Default admin created (admin / admin123)")
        else:
            print("ℹ️ Admin already exists")
    finally:
        db.close()

create_default_admin()

# -------------------- LOAD ML MODEL -------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

model = None
vectorizer = None

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("⚠️ Warning: ML model files missing. /predict endpoint will not work.")
else:
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Model and vectorizer loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load model/vectorizer: {e}")

# Make model accessible to routers
app.state.model = model
app.state.vectorizer = vectorizer

# -------------------- ROUTERS -------------------- #
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(prediction_router)

# -------------------- FRONTEND -------------------- #
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Serve static files (CSS/JS/images)
if os.path.exists(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# Serve index.html at root
@app.get("/")
def serve_frontend():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"error": "Frontend not found"}

# -------------------- HEALTH CHECK -------------------- #
@app.get("/health")
def health():
    return {"status": "Fake Job Detection API running"}
















