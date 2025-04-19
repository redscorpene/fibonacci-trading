#!/usr/bin/env python3
"""
Fibonacci Trading System - Main Entry Point
"""

import os
import time
import logging
import logging.config
from pathlib import Path
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from google.cloud import firestore
from google.cloud import secretmanager
import google.auth
import psutil
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# ========================
# APPLICATION SETUP
# ========================

# Initialize FastAPI with enhanced configuration
app = FastAPI(
    title="Fibonacci Trading API",
    description="AI-powered trading system using Fibonacci retracement patterns",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    debug=os.getenv("DEBUG", "false").lower() == "true"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=json.loads(os.getenv("ALLOWED_ORIGINS", "[\"*\"]")),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# LOGGING CONFIGURATION
# ========================

logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '''
                asctime: %(asctime)s
                name: %(name)s
                levelname: %(levelname)s
                message: %(message)s
                filename: %(filename)s
                lineno: %(lineno)d
                trace_id: %(trace_id)s
                span_id: %(span_id)s
            '''
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/fibonacci-trading.log',
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
            'formatter': 'json'
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': os.getenv("LOG_LEVEL", "INFO")
        },
        'uvicorn.error': {
            'level': 'INFO'
        },
        'uvicorn.access': {
            'level': 'WARNING'
        }
    }
})

logger = logging.getLogger(__name__)

# ========================
# METRICS CONFIGURATION
# ========================

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP Requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

ERROR_COUNT = Counter(
    'http_errors_total',
    'Total HTTP Errors',
    ['method', 'endpoint', 'error_type']
)

API_HEALTH = Gauge(
    'api_health_status',
    'API health status (1=healthy, 0=unhealthy)'
)

# ========================
# MODEL & CONFIG LOADING
# ========================

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
model = None
scaler = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global model, scaler
    
    try:
        # Load models with verification
        model_path = MODEL_DIR / "fibonacci_model_current.pkl"
        scaler_path = MODEL_DIR / "state_scaler.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError("Model files not found")
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Verify model can make predictions
        test_input = np.zeros(16)  # Adjust to your model's input size
        try:
            model.predict([test_input])
            logger.info("Model loaded and verified successfully")
            API_HEALTH.set(1)
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            model = None
            API_HEALTH.set(0)
            raise
            
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}")
        model = None
        scaler = None
        API_HEALTH.set(0)
        raise RuntimeError("Service initialization failed")

# ========================
# DATA MODELS
# ========================

class MarketData(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

class TradingSignal(BaseModel):
    action: str  # "BUY", "SELL", "CLOSE", "HOLD"
    lot_size: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    redraw_fibo: bool = False
    message: str = ""
    trade_id: Optional[str] = None

# ========================
# HELPER FUNCTIONS
# ========================

def prepare_data(candles: List[Dict]) -> pd.DataFrame:
    """Convert and validate candle data with enhanced error handling"""
    try:
        df = pd.DataFrame(candles)
        
        # Validate required columns
        required_cols = ["time", "open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required columns in candle data")
            
        # Convert and clean data
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        df = df.dropna(subset=["open", "high", "low", "close"])
        
        # Calculate technical features
        df["candle_body"] = abs(df["close"] - df["open"])
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["candle_range"] = df["high"] - df["low"]
        
        for n in range(1, 6):
            df[f"breaks_candle_{n}_high"] = (df["close"] > df["high"].shift(n)).astype(int)
            df[f"breaks_candle_{n}_low"] = (df["close"] < df["low"].shift(n)).astype(int)
            
        return df.sort_values("time").drop_duplicates("time")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise ValueError(f"Data processing error: {str(e)}")

def calculate_fibonacci_levels(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate Fibonacci levels with validation and error handling"""
    try:
        if df.empty:
            raise ValueError("Empty DataFrame for Fibonacci calculation")
            
        high_idx = df.iloc[-20:]["high"].idxmax() if len(df) >= 20 else df["high"].idxmax()
        low_idx = df.iloc[-20:]["low"].idxmin() if len(df) >= 20 else df["low"].idxmin()
        
        high_price = df.loc[high_idx, "high"]
        low_price = df.loc[low_idx, "low"]
        price_range = high_price - low_price
        
        if price_range <= 0:
            raise ValueError("Invalid price range for Fibonacci levels")
            
        return {
            "0.0%": high_price,
            "23.6%": high_price - 0.236 * price_range,
            "38.2%": high_price - 0.382 * price_range,
            "50.0%": high_price - 0.5 * price_range,
            "61.8%": high_price - 0.618 * price_range,
            "100.0%": low_price
        }
        
    except Exception as e:
        logger.error(f"Fibonacci calculation failed: {str(e)}")
        raise ValueError(f"Fibonacci calculation error: {str(e)}")

# ========================
# API ENDPOINTS
# ========================

@app.get("/")
async def root():
    """Root endpoint with health information"""
    return {
        "service": "Fibonacci Trading API",
        "status": "operational" if model is not None else "degraded",
        "version": "2.1.0",
        "endpoints": {
            "/analyze": "POST - Analyze market data",
            "/health": "GET - Detailed health check",
            "/metrics": "GET - Prometheus metrics",
            "/docs": "API documentation",
            "/redoc": "Alternative documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    checks = {
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent(),
        "disk_usage": psutil.disk_usage("/").percent,
        "service_status": "healthy" if model is not None else "degraded"
    }
    
    status_code = status.HTTP_200_OK if checks["service_status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=checks)

@app.post("/analyze")
async def analyze_market(request: Request):
    """Main trading analysis endpoint with full error handling"""
    trace_id = request.headers.get("X-Cloud-Trace-Context", "").split("/")[0]
    extra = {"trace_id": trace_id}
    start_time = time.time()
    
    try:
        # Authentication
        try:
            api_key = await get_api_key(request)
            extra["api_key"] = api_key[-4:]  # Log last 4 chars for audit
        except HTTPException as auth_error:
            ERROR_COUNT.labels(method="POST", endpoint="/analyze", error_type="authentication").inc()
            logger.error("Authentication failed", extra=extra)
            raise
        except Exception as auth_error:
            ERROR_COUNT.labels(method="POST", endpoint="/analyze", error_type="authentication").inc()
            logger.error(f"Unexpected auth error: {str(auth_error)}", extra=extra)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication error"
            )

        # Request validation
        try:
            body = await request.json()
            if not body:
                raise ValueError("Empty request body")
                
            required_fields = ["candles", "current_tick", "account"]
            for field in required_fields:
                if field not in body:
                    raise ValueError(f"Missing required field: {field}")
                    
            candles = body["candles"]
            current_tick = body["current_tick"]
            account = body["account"]
            positions = body.get("active_positions", [])
            
            # Validate tick data
            if not isinstance(current_tick.get("bid"), (int, float)) or not isinstance(current_tick.get("ask"), (int, float)):
                raise ValueError("Invalid tick prices")
                
        except ValueError as ve:
            ERROR_COUNT.labels(method="POST", endpoint="/analyze", error_type="validation").inc()
            logger.error(f"Invalid request data: {str(ve)}", extra=extra)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(ve)
            )
        except Exception as e:
            ERROR_COUNT.labels(method="POST", endpoint="/analyze", error_type="validation").inc()
            logger.error(f"Request parsing error: {str(e)}", extra=extra)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request format"
            )

        # Model validation
        if model is None:
            ERROR_COUNT.labels(method="POST", endpoint="/analyze", error_type="unavailable").inc()
            logger.error("Trading model not loaded", extra=extra)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service temporarily unavailable"
            )

        # Data processing
        try:
            df = prepare_data(candles)
            if df.empty:
                raise ValueError("No valid candles after processing")
                
            model_input = create_model_input(df, account, current_tick, positions)
            
            # Validate model input
            if model_input.shape != (16,):  # Adjust based on your model
                raise ValueError(f"Invalid model input shape: {model_input.shape}")
                
        except ValueError as ve:
            ERROR_COUNT.labels(method="POST", endpoint="/analyze", error_type="processing").inc()
            logger.error(f"Data processing error: {str(ve)}", extra=extra)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(ve)
            )
        except Exception as e:
            ERROR_COUNT.labels(method="POST", endpoint="/analyze", error_type="processing").inc()
            logger.error(f"Data processing failed: {str(e)}", extra=extra)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Data processing error"
            )

        # Prediction
        try:
            prediction = model.predict([model_input])[0]
            fib_levels = calculate_fibonacci_levels(df)
            
            # Validate Fibonacci levels
            if not all(isinstance(v, (int, float)) for v in fib_levels.values()):
                raise ValueError("Invalid Fibonacci levels calculated")
                
        except Exception as e:
            ERROR_COUNT.labels(method="POST", endpoint="/analyze", error_type="prediction").inc()
            logger.error(f"Prediction failed: {str(e)}", extra=extra)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction error"
            )

        # Response generation
        try:
            bid_price = float(current_tick["bid"])
            ask_price = float(current_tick["ask"])
            
            signal = TradingSignal(
                action="BUY" if prediction > 0 else "SELL" if prediction < 0 else "HOLD",
                lot_size=np.clip(
                    abs(prediction),
                    float(os.getenv("MIN_LOT_SIZE", "0.01")),
                    float(os.getenv("MAX_LOT_SIZE", "0.1"))
                ),
                entry_price=bid_price if prediction > 0 else ask_price,
                stop_loss=float(fib_levels["0.0%"]),
                take_profit=float(fib_levels["38.2%"]) if prediction > 0 else float(fib_levels["61.8%"]),
                confidence=min(abs(prediction), 1.0),
                message="Analysis completed successfully",
                trade_id=str(uuid.uuid4())
            )
            
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(method="POST", endpoint="/analyze").observe(latency)
            REQUEST_COUNT.labels(method="POST", endpoint="/analyze", status_code="200").inc()
            
            logger.info("Analysis completed successfully", extra={
                **extra,
                "processing_time": latency,
                "prediction": float(prediction),
                "action": signal.action
            })
            
            return signal
            
        except Exception as e:
            ERROR_COUNT.labels(method="POST", endpoint="/analyze", error_type="response").inc()
            logger.error(f"Response generation failed: {str(e)}", extra=extra)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Response generation error"
            )

    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(method="POST", endpoint="/analyze", error_type="unexpected").inc()
        logger.error(f"Unexpected error in analysis: {str(e)}", extra=extra)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# ========================
# METRICS ENDPOINT
# ========================

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type="text/plain")

# ========================
# MAIN ENTRY POINT
# ========================

if __name__ == "__main__":
    import uvicorn
    
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Start FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        log_config=None,  # Use our custom logging config
        timeout_keep_alive=60
    )