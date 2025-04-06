import os
import requests
import time
from datetime import datetime
from google.cloud import firestore
from fastapi import FastAPI
import uvicorn
import logging

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Firestore
db = firestore.Client()

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "market-data-collector"}

@app.get("/collect")
def collect_btcusd_data():
    """Collect 5-minute BTCUSD candle data from Binance public API"""
    try:
        # Use Binance API (free and reliable)
        api_url = "https://api.binance.com/api/v3/klines"
        
        params = {
            "symbol": "BTCUSDT",
            "interval": "5m",
            "limit": 1000
        }
        
        response = requests.get(api_url, params=params)
        data = response.json()
        
        # Process candles
        candles = []
        for candle in data:
            candles.append({
                "time": datetime.fromtimestamp(candle[0]/1000).isoformat(),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])
            })
        
        # Store in Firestore
        doc_ref = db.collection('market_data').document(f'btcusd-{int(time.time())}')
        doc_ref.set({
            'timestamp': datetime.utcnow().isoformat(),
            'candles': candles
        })
        
        logger.info(f"Collected and stored {len(candles)} candles")
        return {"status": "success", "candles_collected": len(candles)}
    except Exception as e:
        logger.error(f"Error collecting market data: {e}")
        return {"status": "error", "message": str(e)}

def scheduled_collection():
    """Background task for scheduled data collection"""
    while True:
        collect_btcusd_data()
        time.sleep(300)  # 5 minutes

@app.on_event("startup")
def startup_event():
    """Start background task when app starts"""
    import threading
    thread = threading.Thread(target=scheduled_collection)
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)