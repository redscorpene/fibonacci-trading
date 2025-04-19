import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback
from google.cloud import firestore, secretmanager
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("service.log")
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fibonacci Trading API",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"]
)

# Initialize services
db = firestore.Client()
secret_client = secretmanager.SecretManagerServiceClient()

# Model loading
MODEL_DIR = Path("/app/models")
model = None
scaler = None

@app.on_event("startup")
async def startup_event():
    global model, scaler
    try:
        model_path = MODEL_DIR / "fibonacci_model_current.pkl"
        scaler_path = MODEL_DIR / "state_scaler.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError("Model files not found")
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Verify model
        test_input = np.zeros(16)
        model.predict([test_input])
        logger.info("Model loaded and verified")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}\n{traceback.format_exc()}")
        raise RuntimeError("Service initialization failed")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        
        # Ensure proper headers
        response.headers["X-Process-Time"] = str(time.time() - start_time)
        response.headers["Connection"] = "keep-alive"
        
        return response
        
    except Exception as e:
        logger.error(f"Request failed: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Service unavailable"},
            headers={"Retry-After": "5"}
        )

@app.post("/analyze")
async def analyze_market(request: Request):
    try:
        data = await request.json()
        
        # Validate required fields
        required = ["candles", "current_tick", "account"]
        if not all(field in data for field in required):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # Process data
        df = pd.DataFrame(data["candles"])
        df["time"] = pd.to_datetime(df["time"])
        
        # Generate prediction
        prediction = model.predict([prepare_features(df)])[0]
        
        return JSONResponse(
            content={
                "status": "success",
                "prediction": float(prediction),
                "timestamp": datetime.now().isoformat()
            },
            headers={
                "Content-Type": "application/json",
                "Cache-Control": "no-cache"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable",
            headers={"Retry-After": "5"}
        )

@app.get("/_health")
async def health_check():
    return JSONResponse(
        content={"status": "ok", "model_loaded": model is not None},
        headers={"Content-Type": "application/json"}
    )

@app.get("/_ready")
async def readiness_check():
    checks = {
        "database": db is not None,
        "model": model is not None,
        "scaler": scaler is not None
    }
    status_code = 200 if all(checks.values()) else 503
    
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready", "checks": checks}
    )

def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Feature engineering logic"""
    # Implement your feature preparation
    return np.zeros(16)  # Replace with actual features