import os
import uuid
import json
import time
import logging
import logging.config
import psutil
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
from google.cloud import firestore
from google.cloud import secretmanager
import google.auth

# Initialize FastAPI with enhanced configuration
app = FastAPI(
    title="Fibonacci Trading AI",
    description="Intelligent trading API for BTC/USD using Fibonacci retracement patterns",
    version="2.0.0",
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

# Configure structured logging
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
        'cloud_logging': {
            'class': 'google.cloud.logging.handlers.CloudLoggingHandler',
            'formatter': 'json'
        },
    },
    'loggers': {
        '': {
            'handlers': ['cloud_logging'],
            'level': 'INFO'
        }
    }
})

logger = logging.getLogger(__name__)

# Initialize services with error handling
try:
    db = firestore.Client()
    logger.info("Firestore client initialized")
except Exception as e:
    logger.error(f"Firestore initialization failed: {str(e)}")
    db = None

try:
    secret_client = secretmanager.SecretManagerServiceClient()
    logger.info("Secret Manager client initialized")
except Exception as e:
    logger.error(f"Secret Manager initialization failed: {str(e)}")
    secret_client = None

# Model loading with verification
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
model = None
scaler = None

@app.on_event("startup")
async def startup_event():
    """Initialize services and verify model loading"""
    global model, scaler
    
    try:
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
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            model = None
            raise
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}")
        model = None
        scaler = None
        raise RuntimeError("Service initialization failed")

# Trading configuration
TRADING_SETTINGS = {
    "points_to_usd_factor": float(os.getenv("POINTS_TO_USD_FACTOR", "0.01")),
    "spread": int(os.getenv("DEFAULT_SPREAD", "3000")),
    "min_lot_size": float(os.getenv("MIN_LOT_SIZE", "0.01")),
    "max_lot_size": float(os.getenv("MAX_LOT_SIZE", "0.1")),
    "max_request_size": int(os.getenv("MAX_REQUEST_SIZE", "10240"))  # 10KB
}

# Data models
class MarketData(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

class AccountInfo(BaseModel):
    balance: float
    equity: float
    margin: Optional[float] = None
    free_margin: Optional[float] = None

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

# Middleware
@app.middleware("http")
async def request_validation_middleware(request: Request, call_next):
    """Validate incoming requests"""
    trace_id = request.headers.get("X-Cloud-Trace-Context", "").split("/")[0]
    span_id = request.headers.get("X-Cloud-Trace-Context", "").split("/")[1] if "/" in request.headers.get("X-Cloud-Trace-Context", "") else ""
    
    extra = {
        "trace_id": trace_id,
        "span_id": span_id,
        "request_path": request.url.path
    }
    
    try:
        # Skip validation for docs and health checks
        if request.url.path in ["/docs", "/redoc", "/health"]:
            return await call_next(request)
            
        # Check content type
        if request.method == "POST" and request.headers.get("content-type") != "application/json":
            logger.warning("Invalid content type", extra=extra)
            return JSONResponse(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                content={"detail": "Unsupported Media Type"}
            )
            
        # Check body size
        body = await request.body()
        if len(body) > TRADING_SETTINGS["max_request_size"]:
            logger.warning("Request too large", extra=extra)
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Payload too large"}
            )
            
        request.state.raw_body = body
        response = await call_next(request)
        return response
        
    except Exception as e:
        logger.error(f"Request validation failed: {str(e)}", extra=extra)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Invalid request"}
        )

# Helper functions
def prepare_data(candles: List[Dict]) -> pd.DataFrame:
    """Convert and validate candle data"""
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
        raise

def create_model_input(df: pd.DataFrame, account: Dict, current_tick: Dict, positions: List[Dict]) -> np.ndarray:
    """Create validated model input features"""
    try:
        if df.empty:
            raise ValueError("Empty DataFrame provided")
            
        latest_candle = df.iloc[-1]
        prev_candles = df.iloc[max(0, len(df)-10):-1]
        
        features = [
            latest_candle["close"] / latest_candle["open"] - 1,
            latest_candle["high"] / latest_candle["low"] - 1,
            latest_candle["candle_body"] / latest_candle["candle_range"] if latest_candle["candle_range"] != 0 else 0,
            prev_candles["candle_range"].mean() if len(prev_candles) > 0 else latest_candle["candle_range"],
            prev_candles["candle_range"].std() if len(prev_candles) > 1 else 0,
            1 if any(latest_candle[f"breaks_candle_{n}_high"] == 1 for n in range(3, 6)) else 0,
            1 if any(latest_candle[f"breaks_candle_{n}_low"] == 1 for n in range(3, 6)) else 0,
            latest_candle["volume"] / prev_candles["volume"].mean() if "volume" in latest_candle and len(prev_candles) > 0 and prev_candles["volume"].mean() > 0 else 1.0,
            pd.to_datetime(latest_candle["time"]).hour / 24.0,
            account.get("balance", 1000) / 10000,
            1 if positions else 0,
            (datetime.now() - pd.to_datetime(positions[0].get("timestamp"))).total_seconds() / 86400 if positions else 0,
            1 if positions and positions[0].get("fib_redrawn", False) else 0,
            1.0,
            (latest_candle["close"] - prev_candles.iloc[0]["close"]) / prev_candles.iloc[0]["close"] if len(prev_candles) >= 10 else 0,
            min(current_tick.get("spread", 0) / (prev_candles["candle_range"].mean() if len(prev_candles) > 0 else latest_candle["candle_range"]), 1) if (prev_candles["candle_range"].mean() if len(prev_candles) > 0 else latest_candle["candle_range"]) > 0 else 0
        ]
        
        return np.array(features)
        
    except Exception as e:
        logger.error(f"Feature creation failed: {str(e)}")
        raise

def calculate_fibonacci_levels(df: pd.DataFrame, broken_candle_idx: int = None) -> Dict[str, float]:
    """Calculate Fibonacci levels with validation"""
    try:
        if df.empty:
            raise ValueError("Empty DataFrame for Fibonacci calculation")
            
        if broken_candle_idx is None:
            high_idx = df.iloc[-20:]["high"].idxmax() if len(df) >= 20 else df["high"].idxmax()
            low_idx = df.iloc[-20:]["low"].idxmin() if len(df) >= 20 else df["low"].idxmin()
        else:
            high_idx = broken_candle_idx
            low_idx = broken_candle_idx
            
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
        raise

async def get_api_key(request: Request) -> str:
    """Validate API key with enhanced security"""
    try:
        body = await request.json()
        api_key = body.get("api_key")
        
        if not api_key:
            raise ValueError("Missing API key")
            
        # Get valid key from Secret Manager
        try:
            secret_name = f"projects/{os.environ['GCP_PROJECT_ID']}/secrets/api-key/versions/latest"
            response = secret_client.access_secret_version(request={"name": secret_name})
            valid_key = response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.error(f"Secret Manager access failed: {str(e)}")
            valid_key = os.environ.get("API_KEY")
            
        if api_key != valid_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
            
        return api_key
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API key validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication error"
        )

# API endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Fibonacci Trading API",
        "status": "operational" if model is not None else "degraded",
        "version": "2.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze market data",
            "/health": "GET - Detailed health check",
            "/docs": "API documentation",
            "/redoc": "Alternative documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    checks = {
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "database_connected": db is not None,
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent(),
        "disk_usage": psutil.disk_usage("/").percent,
        "service_status": "healthy" if model is not None and scaler is not None else "degraded"
    }
    
    status_code = status.HTTP_200_OK if checks["service_status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=status_code,
        content=checks
    )

@app.post("/analyze")
async def analyze_market(request: Request):
    """Analyze market data with comprehensive error handling"""
    trace_id = request.headers.get("X-Cloud-Trace-Context", "").split("/")[0]
    extra = {"trace_id": trace_id}
    
    try:
        # Authentication
        try:
            await get_api_key(request)
        except HTTPException as auth_error:
            logger.error(f"Authentication failed", extra=extra)
            raise
        except Exception as auth_error:
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
            logger.error(f"Invalid request data: {str(ve)}", extra=extra)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(ve)
            )
        except Exception as e:
            logger.error(f"Request parsing error: {str(e)}", extra=extra)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request format"
            )

        # Model validation
        if model is None:
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
            logger.error(f"Data processing error: {str(ve)}", extra=extra)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(ve)
            )
        except Exception as e:
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
                    TRADING_SETTINGS["min_lot_size"],
                    TRADING_SETTINGS["max_lot_size"]
                ),
                entry_price=bid_price if prediction > 0 else ask_price,
                stop_loss=float(fib_levels["0.0%"]),
                take_profit=float(fib_levels["38.2%"]) if prediction > 0 else float(fib_levels["61.8%"]),
                confidence=min(abs(prediction), 1.0),
                message="Analysis completed successfully",
                trade_id=str(uuid.uuid4())
            )
            
            logger.info("Analysis completed successfully", extra=extra)
            return signal
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}", extra=extra)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Response generation error"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analysis: {str(e)}", extra=extra)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )