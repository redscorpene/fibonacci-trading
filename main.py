import os
import uuid
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
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

# Get the current directory (root directory)
current_dir = Path(__file__).parent

# Load model and scaler with proper path handling
try:
    model_path = current_dir / 'fibonacci_model.pkl'
    scaler_path = current_dir / 'state_scaler.pkl'
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    raise RuntimeError(f"Failed to load model files: {e}")

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

class TickData(BaseModel):
    bid: float
    ask: float
    time: str
    spread: float

class PositionData(BaseModel):
    ticket: int
    type: str
    volume: float
    open_price: float
    sl: float
    tp: float
    profit: float
    fib_redrawn: bool
    broken_candle_high: Optional[float] = None
    broken_candle_low: Optional[float] = None
    fib_0: Optional[float] = None
    trade_id: Optional[str] = None
    timestamp: Optional[str] = None
    
class TradingRequest(BaseModel):
    api_key: str
    candles: List[Dict]  # Changed to Dict to handle raw JSON
    current_tick: Dict   # Changed to Dict to handle raw JSON
    account: Dict        # Changed to Dict to handle raw JSON
    active_positions: List[Dict] = []
    
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
async def get_api_key(request: Request):
    """Extract and validate API key from request body (for JSON payload)"""
    try:
        body = await request.json()
        api_key = body.get('api_key')
        
        # Get valid key from Secret Manager or environment
        try:
            secret_name = f"projects/{os.environ['GCP_PROJECT_ID']}/secrets/api-key/versions/latest"
            response = secret_client.access_secret_version(request={"name": secret_name})
            valid_key = response.payload.data.decode("UTF-8")
        except Exception:
            # Fallback for development
            valid_key = os.environ.get('API_KEY', 'dev-key')
            
        if api_key != valid_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
            
        return api_key
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"API key validation error: {str(e)}")

# Helper functions
def prepare_data(candles: List[Dict]) -> pd.DataFrame:
    """Convert candle data to DataFrame"""
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    df['candle_body'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['candle_range'] = df['high'] - df['low']
    
    for n in range(1, 6):
        df[f'breaks_candle_{n}_high'] = (df['close'] > df['high'].shift(n)).astype(int)
        df[f'breaks_candle_{n}_low'] = (df['close'] < df['low'].shift(n)).astype(int)
    
    return df

def create_model_input(df: pd.DataFrame, account: Dict, current_tick: Dict, positions: List[Dict]) -> np.ndarray:
    """Create model input features"""
    latest_candle = df.iloc[-1]
    prev_candles = df.iloc[max(0, len(df)-10):-1]
    
    # Calculate basic features
    features = [
        latest_candle['close'] / latest_candle['open'] - 1,
        latest_candle['high'] / latest_candle['low'] - 1,
        latest_candle['candle_body'] / latest_candle['candle_range'] if latest_candle['candle_range'] != 0 else 0,
        prev_candles['candle_range'].mean() if len(prev_candles) > 0 else latest_candle['candle_range'],
        prev_candles['candle_range'].std() if len(prev_candles) > 1 else 0,
        1 if any([latest_candle[f'breaks_candle_{n}_high'] == 1 for n in range(3, 6)]) else 0,
        1 if any([latest_candle[f'breaks_candle_{n}_low'] == 1 for n in range(3, 6)]) else 0
    ]
    
    # Add volume info if available
    if 'volume' in latest_candle and len(prev_candles) > 0 and prev_candles['volume'].mean() > 0:
        features.append(latest_candle['volume'] / prev_candles['volume'].mean())
    else:
        features.append(1.0)  # Default value
        
    # Time of day feature (normalized hour)
    hour = pd.to_datetime(latest_candle['time']).hour
    features.append(hour / 24.0)
    
    # Account state
    features.append(account.get('balance', 1000) / 10000)  # Normalized balance
        
    # Position info
    has_open_position = len(positions) > 0
    features.append(1 if has_open_position else 0)
    
    # Position duration if any
    position_duration = 0
    if has_open_position:
        try:
            pos_time = pd.to_datetime(positions[0].get('timestamp', datetime.now().isoformat()))
            duration_mins = (datetime.now() - pos_time).total_seconds() / 60
            position_duration = min(duration_mins / 1440, 1.0)  # Normalize to max 1 day
        except:
            position_duration = 0
    
    features.append(position_duration)
    
    # Fib redrawn status
    features.append(1 if has_open_position and positions[0].get('fib_redrawn', False) else 0)
    
    # Target metric (simplified)
    features.append(1.0)
    
    # Market trend (simple)
    if len(prev_candles) >= 10:
        trend = (latest_candle['close'] - prev_candles.iloc[0]['close']) / prev_candles.iloc[0]['close']
        features.append(min(max(trend, -1), 1))  # Clip to [-1, 1]
    else:
        features.append(0)
        
    # Spread factor
    spread = current_tick.get('spread', 0)
    avg_candle_range = prev_candles['candle_range'].mean() if len(prev_candles) > 0 else latest_candle['candle_range']
    features.append(min(spread / avg_candle_range, 1) if avg_candle_range > 0 else 0)
    
    return np.array(features)

def calculate_fibonacci_levels(df: pd.DataFrame, broken_candle_idx: int = None):
    """Calculate Fibonacci retracement levels"""
    # Find significant high/low for retracement
    if broken_candle_idx is None:
        # Find candle with highest high in last 20 candles
        high_idx = df.iloc[-20:]['high'].idxmax() if len(df) >= 20 else df['high'].idxmax()
        # Find candle with lowest low in last 20 candles
        low_idx = df.iloc[-20:]['low'].idxmin() if len(df) >= 20 else df['low'].idxmin()
    else:
        high_idx = broken_candle_idx
        low_idx = broken_candle_idx
    
    high_price = df.loc[high_idx, 'high']
    low_price = df.loc[low_idx, 'low']
    price_range = high_price - low_price
    
    # Get Fibonacci levels as a dict
    levels = {
        "0.0": low_price,
        "0.236": low_price + 0.236 * price_range,
        "0.382": low_price + 0.382 * price_range,
        "0.5": low_price + 0.5 * price_range,
        "0.618": low_price + 0.618 * price_range,
        "0.786": low_price + 0.786 * price_range,
        "1.0": high_price,
        "1.272": high_price + 0.272 * price_range,
        "1.618": high_price + 0.618 * price_range
    }
    
    return levels, high_idx, low_idx

def log_trade_decision(decision: Dict, request_data: Dict):
    """Log trading decision to Firestore"""
    try:
        trade_ref = db.collection('trading_decisions').document()
        trade_data = {
            **decision,
            'account_info': request_data.get('account', {}),
            'timestamp': datetime.utcnow().isoformat(),
            'candle_data': request_data.get('candles', [])[-1] if request_data.get('candles') else {}
        }
        trade_ref.set(trade_data)
    except Exception as e:
        print(f"Error logging trade: {e}")

# API Endpoints
@app.post("/analyze")
async def analyze_market(request: Request, background_tasks: BackgroundTasks):
    """Analyze market data and return trading decision"""
    try:
        # Parse JSON directly from request body
        body = await request.body()
        request_data = json.loads(body.decode())
        
        # Validate API key
        api_key = request_data.get('api_key')
        if api_key != os.environ.get('API_KEY', 'dev-key'):
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Prepare data and get prediction
        df = prepare_data(request_data['candles'])
        features = create_model_input(
            df, 
            request_data['account'], 
            request_data['current_tick'],
            request_data.get('active_positions', [])
        )
        
        # Make prediction if model is available
        if model is not None:
            features_normalized = scaler.transform(features.reshape(1, -1))
            
            # For ensemble models
            if hasattr(model, '__iter__'):
                prediction = np.mean([m.predict(features_normalized)[0] for m in model], axis=0)
            else:
                prediction = model.predict(features_normalized)[0]
                
            # Use prediction to determine action
            confidence = float(np.max(prediction))
            
            # Determine trade action
            action = "HOLD"
            lot_size = 0.0
            entry_price = None
            stop_loss = None
            take_profit = None
            message = "No trading opportunity detected"
            redraw_fibo = False
            
            # Simple example decision logic (implement your full logic)
            if confidence > 0.7:
                # Check for broken candle pattern
                broken_high = any([df.iloc[-1][f'breaks_candle_{n}_high'] == 1 for n in range(2, 5)])
                broken_low = any([df.iloc[-1][f'breaks_candle_{n}_low'] == 1 for n in range(2, 5)])
                
                current_price = request_data['current_tick']['bid']
                
                if broken_high:
                    action = "BUY"
                    message = "Bullish breakout detected"
                    
                    # Calculate Fibonacci retracement levels for this trade
                    fib_levels, _, _ = calculate_fibonacci_levels(df)
                    
                    # Set entry, SL and TP
                    entry_price = current_price
                    stop_loss = fib_levels["0.5"]  # Use 50% retracement as SL
                    take_profit = fib_levels["1.272"]  # Use 127.2% extension as TP
                    
                    # Calculate position size (simplified)
                    account_balance = request_data['account']['balance']
                    risk_pct = min(0.02, prediction[0])  # Risk up to 2% of account
                    risk_amount = account_balance * risk_pct
                    
                    # Calculate position size based on SL distance
                    sl_distance = abs(entry_price - stop_loss)
                    if sl_distance > 0:
                        lot_size = max(min(risk_amount / sl_distance, TRADING_SETTINGS["max_lot_size"]), 
                                      TRADING_SETTINGS["min_lot_size"])
                    else:
                        lot_size = TRADING_SETTINGS["min_lot_size"]
                        
                elif broken_low:
                    action = "SELL"
                    message = "Bearish breakdown detected"
                    
                    # Calculate Fibonacci retracement levels for this trade
                    fib_levels, _, _ = calculate_fibonacci_levels(df)
                    
                    # Set entry, SL and TP
                    entry_price = current_price
                    stop_loss = fib_levels["0.5"]  # Use 50% retracement as SL
                    take_profit = fib_levels["0.0"]  # Use 0% retracement as TP
                    
                    # Calculate position size (simplified)
                    account_balance = request_data['account']['balance']
                    risk_pct = min(0.02, prediction[0])  # Risk up to 2% of account
                    risk_amount = account_balance * risk_pct
                    
                    # Calculate position size based on SL distance
                    sl_distance = abs(entry_price - stop_loss)
                    if sl_distance > 0:
                        lot_size = max(min(risk_amount / sl_distance, TRADING_SETTINGS["max_lot_size"]), 
                                      TRADING_SETTINGS["min_lot_size"])
                    else:
                        lot_size = TRADING_SETTINGS["min_lot_size"]
        else:
            # Default response when no model is available
            action = "HOLD"
            lot_size = 0.0
            entry_price = None
            stop_loss = None
            take_profit = None
            confidence = 0.0
            message = "Model not available"
            redraw_fibo = False
            
        # Create trading signal
        trade_id = str(uuid.uuid4())
        signal = {
            "action": action,
            "lot_size": float(lot_size),
            "entry_price": float(entry_price) if entry_price is not None else None,
            "stop_loss": float(stop_loss) if stop_loss is not None else None,
            "take_profit": float(take_profit) if take_profit is not None else None,
            "confidence": float(confidence),
            "redraw_fibo": redraw_fibo,
            "message": message,
            "trade_id": trade_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log the decision
        background_tasks.add_task(log_trade_decision, signal, request_data)
        
        return signal
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing market: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Verify connections
        test_doc = db.collection('health').document('check')
        test_doc.set({'timestamp': datetime.utcnow().isoformat()})
        
        return {
            "status": "healthy",
            "services": {
                "firestore": "connected",
                "model": "loaded" if model is not None else "not loaded",
                "scaler": "loaded" if scaler is not None else "not loaded"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service unavailable: {str(e)}")

@app.post("/trade-result")
async def record_trade_result(request: Request):
    """Record trade result for continuous learning"""
    try:
        body = await request.body()
        result_data = json.loads(body.decode())
        
        # Validate required fields
        if 'trade_id' not in result_data:
            raise HTTPException(status_code=400, detail="Missing trade_id")
            
        # Record to Firestore
        result_ref = db.collection('trade_results').document()
        result_ref.set({
            **result_data,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return {"status": "success", "message": "Trade result recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording trade result: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)