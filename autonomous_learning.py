import pandas as pd
import numpy as np
from google.cloud import firestore
from datetime import datetime, timedelta
import joblib
import os
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Firestore
db = firestore.Client()

def get_recent_market_data(hours=24):
    """Get recent market data from Firestore"""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    try:
        docs = db.collection('market_data').where(
            'timestamp', '>=', cutoff_time.isoformat()
        ).stream()
        
        all_candles = []
        for doc in docs:
            data = doc.to_dict()
            all_candles.extend(data.get('candles', []))
        
        # Convert to DataFrame and sort by time
        if all_candles:
            df = pd.DataFrame(all_candles)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').drop_duplicates('time')
            logger.info(f"Retrieved {len(df)} candles from the last {hours} hours")
            return df
        else:
            logger.warning(f"No candles found in the last {hours} hours")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error retrieving market data: {e}")
        return pd.DataFrame()

def prepare_data(df):
    """Process raw data for pattern detection"""
    # Calculate features
    df['candle_body'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['candle_range'] = df['high'] - df['low']
    
    # Identify breakout patterns
    for n in range(1, 6):
        df[f'breaks_candle_{n}_high'] = (df['close'] > df['high'].shift(n)).astype(int)
        df[f'breaks_candle_{n}_low'] = (df['close'] < df['low'].shift(n)).astype(int)
    
    # Add pattern flags
    df['breaks_pattern_bullish'] = ((df['breaks_candle_3_high'] == 1) | 
                                    (df['breaks_candle_4_high'] == 1) | 
                                    (df['breaks_candle_5_high'] == 1)).astype(int)
    
    df['breaks_pattern_bearish'] = ((df['breaks_candle_3_low'] == 1) | 
                                    (df['breaks_candle_4_low'] == 1) | 
                                    (df['breaks_candle_5_low'] == 1)).astype(int)
    
    return df

def analyze_patterns(df):
    """Find and analyze Fibonacci patterns"""
    patterns = []
    
    for i in range(10, len(df) - 50):
        candle = df.iloc[i]
        
        bullish_break = any([candle[f'breaks_candle_{n}_high'] == 1 for n in range(3, 6)])
        bearish_break = any([candle[f'breaks_candle_{n}_low'] == 1 for n in range(3, 6)])
        
        if bullish_break or bearish_break:
            # Find broken candle
            broken_idx = None
            for n in range(3, 6):
                if bullish_break and candle[f'breaks_candle_{n}_high'] == 1:
                    broken_idx = i - n
                    break
                elif bearish_break and candle[f'breaks_candle_{n}_low'] == 1:
                    broken_idx = i - n
                    break
            
            if broken_idx is not None:
                # Calculate Fibonacci levels
                if bullish_break:
                    fib_high = candle['close']
                    fib_low = df.iloc[broken_idx]['low']
                    pattern_type = 'bullish'
                else:
                    fib_high = df.iloc[broken_idx]['high']
                    fib_low = candle['close']
                    pattern_type = 'bearish'
                
                fib_range = fib_high - fib_low
                
                if fib_range > 0:
                    if pattern_type == 'bullish':
                        fib_0 = fib_high
                        fib_32 = fib_high - 0.322 * fib_range
                        fib_161 = fib_high + 1.618 * fib_range
                    else:
                        fib_0 = fib_low
                        fib_32 = fib_low + 0.322 * fib_range
                        fib_161 = fib_low - 1.618 * fib_range
                    
                    # Track what happens after pattern
                    sl_hit = False
                    tp_hit = False
                    final_price = candle['close']
                    
                    for j in range(1, min(50, len(df) - i - 1)):
                        future_candle = df.iloc[i + j]
                        
                        if pattern_type == 'bullish':
                            if future_candle['low'] <= fib_32:
                                sl_hit = True
                                final_price = fib_32
                                break
                            elif future_candle['high'] >= fib_161:
                                tp_hit = True
                                final_price = fib_161
                                break
                        else:
                            if future_candle['high'] >= fib_32:
                                sl_hit = True
                                final_price = fib_32
                                break
                            elif future_candle['low'] <= fib_161:
                                tp_hit = True
                                final_price = fib_161
                                break
                        
                        # Update final price as we go
                        final_price = future_candle['close']
                    
                    # Calculate profit/loss
                    if pattern_type == 'bullish':
                        profit = final_price - candle['close']
                    else:
                        profit = candle['close'] - final_price
                    
                    # Store pattern info
                    patterns.append({
                        'time': candle['time'].isoformat(),
                        'pattern_type': pattern_type,
                        'fib_range': float(fib_range),
                        'fib_0': float(fib_0),
                        'fib_32': float(fib_32), 
                        'fib_161': float(fib_161),
                        'sl_hit': sl_hit,
                        'tp_hit': tp_hit,
                        'profit': float(profit),
                        'entry_price': float(candle['close']),
                        'exit_price': float(final_price),
                        'success': tp_hit,
                        'timestamp': datetime.utcnow().isoformat()
                    })
    
    return patterns

def update_learning_data(patterns):
    """Update learning database with new pattern results"""
    for pattern in patterns:
        try:
            doc_ref = db.collection('learning_data').document()
            doc_ref.set(pattern)
        except Exception as e:
            logger.error(f"Error storing pattern data: {e}")
    
    logger.info(f"Stored {len(patterns)} patterns in learning database")

def update_model(model_path="/app/models/fibonacci_model.pkl"):
    """Update model based on historical pattern performance"""
    try:
        # Load model
        if os.path.exists(model_path):
            models = joblib.load(model_path)
            logger.info("Loaded existing model")
        else:
            logger.warning("Model not found, skipping update")
            return False
        
        # Get learning data
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        docs = db.collection('learning_data').where(
            'timestamp', '>=', cutoff_time.isoformat()
        ).stream()
        
        learning_data = []
        for doc in docs:
            data = doc.to_dict()
            learning_data.append(data)
        
        if not learning_data:
            logger.warning("No learning data available, skipping update")
            return False
        
        logger.info(f"Found {len(learning_data)} learning data points")
        
        # Extract features and targets
        X = []
        y = []
        
        # Here you would create similar features as in your trading model
        # This is a simplified example
        for data in learning_data:
            # Your feature extraction logic here
            pass
        
        # If we have enough data, update model
        if len(X) >= 10:
            # Update model with new data
            # Your model update logic here
            
            # Save updated model
            joblib.dump(models, model_path)
            logger.info("Model updated and saved")
            return True
        else:
            logger.warning("Not enough data points for model update")
            return False
            
    except Exception as e:
        logger.error(f"Error updating model: {e}")
        return False

def main():
    logger.info("Starting autonomous learning system")
    
    while True:
        try:
            # Get recent market data
            logger.info("Fetching market data...")
            df = get_recent_market_data(hours=48)
            
            if not df.empty:
                # Process data
                logger.info("Processing data...")
                df = prepare_data(df)
                
                # Analyze patterns
                logger.info("Analyzing patterns...")
                patterns = analyze_patterns(df)
                
                # Update learning data
                if patterns:
                    logger.info(f"Found {len(patterns)} patterns to analyze")
                    update_learning_data(patterns)
                    
                    # Try to update model (every 12 hours)
                    current_hour = datetime.now().hour
                    if current_hour % 12 == 0:
                        logger.info("Attempting model update...")
                        update_model()
                else:
                    logger.info("No patterns found in recent data")
            else:
                logger.warning("No market data available")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        
        # Sleep for an hour before next analysis
        logger.info("Sleeping for one hour...")
        time.sleep(3600)

if __name__ == "__main__":
    main()