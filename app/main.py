import os
import uuid
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from google.cloud import firestore
from google.cloud import secretmanager
import google.auth

# Initialize FastAPI with documentation endpoints
app = FastAPI(
    title="Fibonacci Trading AI",
    description="Intelligent trading API for BTC/USD using Fibonacci retracement patterns",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = joblib.load('/app/app/models/fibonacci_model.pkl')
scaler = joblib.load("/app/app/models/state_scaler.pkl")

# Initialize Firestore
db = firestore.Client()

# Initialize Secret Manager
secret_client = secretmanager.SecretManagerServiceClient()

# Trading settings
TRADING_SETTINGS = {
    "points_to_usd_factor": 1.25/124.9,
    "spread": 3000,
    "min_lot_size": 0.01,
    "max_lot_size": 0.1
}

# Model classes
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
    
class TradingRequest(BaseModel):
    api_key: str
    candles: List[MarketData]
    account: AccountInfo
    active_positions: List[Dict] = []
    
class TradingSignal(BaseModel):
    action: str  # "BUY", "SELL", "CLOSE", "HOLD"
    lot_size: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float
    redraw_fibo: bool = False
    message: str

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Fibonacci Trading API", 
        "endpoints": {
            "/analyze": "POST - Analyze market data",
            "/health": "GET - Health check",
            "/docs": "API documentation",
            "/redoc": "Alternative documentation"
        },
        "status": "operational"
    }

# API key validation
def get_api_key(api_key: str = Header(..., alias="api-key")):
    """Validate API key using Google Secret Manager"""
    try:
        secret_name = f"projects/{os.environ['GCP_PROJECT_ID']}/secrets/api-key/versions/latest"
        response = secret_client.access_secret_version(request={"name": secret_name})
        valid_key = response.payload.data.decode("UTF-8")
        
        if api_key != valid_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    except Exception as e:
        # Fallback for development
        if api_key != os.environ.get('API_KEY', 'dev-key'):
            raise HTTPException(status_code=401, detail=f"Invalid API key: {str(e)}")
    return api_key

# Helper functions
def prepare_data(candles: List[MarketData]) -> pd.DataFrame:
    """Convert candle data to DataFrame"""
    df = pd.DataFrame([c.dict() for c in candles])
    df['time'] = pd.to_datetime(df['time'])
    df['candle_body'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['candle_range'] = df['high'] - df['low']
    
    for n in range(1, 6):
        df[f'breaks_candle_{n}_high'] = (df['close'] > df['high'].shift(n)).astype(int)
        df[f'breaks_candle_{n}_low'] = (df['close'] < df['low'].shift(n)).astype(int)
    
    return df

def create_model_input(df: pd.DataFrame, account: AccountInfo) -> np.ndarray:
    """Create model input features"""
    latest_candle = df.iloc[-1]
    prev_candles = df.iloc[max(0, len(df)-10):-1]
    
    features = np.array([
        latest_candle['close'] / latest_candle['open'] - 1,
        latest_candle['high'] / latest_candle['low'] - 1,
        latest_candle['candle_body'] / latest_candle['candle_range'] if latest_candle['candle_range'] != 0 else 0,
        prev_candles['candle_range'].mean() if len(prev_candles) > 0 else latest_candle['candle_range'],
        prev_candles['candle_range'].std() if len(prev_candles) > 1 else 0,
        1 if any([latest_candle[f'breaks_candle_{n}_high'] == 1 for n in range(3, 6)]) else 0,
        1 if any([latest_candle[f'breaks_candle_{n}_low'] == 1 for n in range(3, 6)]) else 0,
        latest_candle['volume'] / prev_candles['volume'].mean() if 'volume' in latest_candle and len(prev_candles) > 0 and prev_cand

    def log_trade_decision(decision: TradingSignal, request_data: TradingRequest):
        """Log trading decision to Firestore"""
        try:
            trade_ref = db.collection('trading_decisions').document()
            trade_data = {
                **decision.dict(),
                'account_info': request_data.account.dict(),
                'timestamp': datetime.utcnow().isoformat(),
                'candle_data': request_data.candles[-1].dict()
            }
            trade_ref.set(trade_data)
        except Exception as e:
            print(f"Error logging trade: {e}")

    # API Endpoints
    @app.post("/analyze", response_model=TradingSignal, dependencies=[Depends(get_api_key)])
    async def analyze_market(request: TradingRequest, background_tasks: BackgroundTasks):
        """Analyze market data and return trading decision"""
        try:
            # Prepare data and get prediction
            df = prepare_data(request.candles)
            features = create_model_input(df, request.account)
            features_normalized = scaler.transform(features.reshape(1, -1))
            action = model.predict(features_normalized)[0]
            
            # Generate trading signal (simplified for example)
            signal = TradingSignal(
                action="HOLD",
                lot_size=0.0,
                confidence=0.0,
                message="Sample signal - implement your logic here"
            )
            
            # Log the decision
            background_tasks.add_task(log_trade_decision, signal, request)
            return signal
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing market: {str(e)}")

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        try:
            # Verify Firestore connection
            db.collection('health').document('check').set({'timestamp': datetime.utcnow().isoformat()})
            return {
                "status": "healthy",
                "services": {
                    "firestore": "connected",
                    "model": "loaded",
                    "scaler": "loaded"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Service unavailable: {str(e)}")

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)