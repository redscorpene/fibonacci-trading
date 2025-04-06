import requests
import pandas as pd
import time
import json
from datetime import datetime
import os
from google.cloud import firestore
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Firestore
db = firestore.Client()

def collect_btcusd_data():
    """Collect 5-minute BTCUSD candle data from Binance public API"""
    try:
        # Use Binance API (free and reliable)
        api_url = "https://api.binance.com/api/v3/klines"
        
        params = {
            "symbol": "BTCUSDT",  # Binance uses BTCUSDT
            "interval": "5m",
            "limit": 1000         # Get maximum allowed candles
        }
        
        response = requests.get(api_url, params=params)
        data = response.json()
        
        # Process candles
        candles = []
        for candle in data:
            # Binance kline format: [Open time, Open, High, Low, Close, Volume, ...]
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
        return candles
    except Exception as e:
        logger.error(f"Error collecting market data: {e}")
        return []

def main():
    logger.info("Starting BTC/USD market data collection service")
    while True:
        collect_btcusd_data()
        # Wait 5 minutes before next collection
        time.sleep(300)

if __name__ == "__main__":
    main()