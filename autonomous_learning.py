import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fibonacci_autonomous')

# Initialize Firestore with error handling
db = None
try:
    from google.cloud import firestore
    db = firestore.Client()
    logger.info("Firestore client initialized successfully")
except ImportError:
    logger.error("Google Cloud Firestore library not available")
except Exception as e:
    logger.error(f"Failed to initialize Firestore: {str(e)}")

# Model directory configuration
MODEL_DIR = Path(os.getenv('MODEL_DIR', '/app/models'))
if not MODEL_DIR.exists():
    MODEL_DIR = Path(__file__).parent / 'models'
    MODEL_DIR.mkdir(exist_ok=True)
    logger.info(f"Created model directory at {MODEL_DIR}")

def get_recent_market_data(hours: int = 24) -> pd.DataFrame:
    """Retrieve market data from Firestore with robust error handling"""
    if db is None:
        logger.error("Cannot retrieve data - Firestore not initialized")
        return pd.DataFrame()
    
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    try:
        docs = db.collection('market_data').where(
            'timestamp', '>=', cutoff_time
        ).order_by('timestamp').stream()
        
        all_candles = []
        for doc in docs:
            data = doc.to_dict()
            if 'candles' in data and isinstance(data['candles'], list):
                all_candles.extend(data['candles'])
        
        if not all_candles:
            logger.warning(f"No candles found in last {hours} hours")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_candles)
        
        # Data validation and cleaning
        required_cols = ['time', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error("Missing required columns in market data")
            return pd.DataFrame()
            
        # Convert and clean data
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])
        df = df.sort_values('time').drop_duplicates('time')
        
        # Convert numeric columns
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        logger.info(f"Retrieved {len(df)} valid candles")
        return df
        
    except Exception as e:
        logger.error(f"Error retrieving market data: {str(e)}")
        return pd.DataFrame()

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enhance market data with technical features"""
    if df.empty:
        return df
    
    try:
        # Calculate basic candle metrics
        df['candle_body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']
        df['body_ratio'] = df['candle_body'] / df['candle_range'].replace(0, 1e-10)
        
        # Breakout patterns
        for n in range(1, 6):
            df[f'breaks_high_{n}'] = (df['close'] > df['high'].shift(n)).astype(int)
            df[f'breaks_low_{n}'] = (df['close'] < df['low'].shift(n)).astype(int)
        
        # Pattern flags
        df['bullish_break'] = df[[f'breaks_high_{n}' for n in [3,4,5]]].max(axis=1)
        df['bearish_break'] = df[[f'breaks_low_{n}' for n in [3,4,5]]].max(axis=1)
        
        # Rolling metrics
        df['volatility'] = df['candle_range'].rolling(14).std()
        df['volume_ma'] = df['volume'].rolling(14).mean() if 'volume' in df.columns else 0
        
        return df
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        return pd.DataFrame()

def analyze_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify and analyze Fibonacci patterns in market data"""
    patterns = []
    
    if df.empty or len(df) < 20:
        logger.warning("Insufficient data for pattern analysis")
        return patterns
    
    try:
        for i in range(10, len(df)-20):
            candle = df.iloc[i]
            
            # Check for valid breakout patterns
            bullish = candle['bullish_break'] == 1
            bearish = candle['bearish_break'] == 1
            
            if not (bullish or bearish):
                continue
                
            # Find broken candle
            broken_idx = None
            for n in [3,4,5]:
                if bullish and candle[f'breaks_high_{n}'] == 1:
                    broken_idx = i - n
                    break
                elif bearish and candle[f'breaks_low_{n}'] == 1:
                    broken_idx = i - n
                    break
            
            if broken_idx is None or broken_idx < 0:
                continue
                
            # Calculate Fibonacci levels
            if bullish:
                swing_high = candle['close']
                swing_low = df.iloc[broken_idx]['low']
                pattern_type = 'bullish'
            else:
                swing_high = df.iloc[broken_idx]['high']
                swing_low = candle['close']
                pattern_type = 'bearish'
            
            price_range = swing_high - swing_low
            if price_range <= 0:
                continue
                
            # Calculate key Fibonacci levels
            levels = {
                '0.0': swing_high if bullish else swing_low,
                '0.236': swing_high - 0.236 * price_range if bullish else swing_low + 0.236 * price_range,
                '0.382': swing_high - 0.382 * price_range if bullish else swing_low + 0.382 * price_range,
                '0.5': swing_high - 0.5 * price_range if bullish else swing_low + 0.5 * price_range,
                '0.618': swing_high - 0.618 * price_range if bullish else swing_low + 0.618 * price_range,
                '1.0': swing_low if bullish else swing_high,
                '1.618': swing_high + 0.618 * price_range if bullish else swing_low - 0.618 * price_range
            }
            
            # Track pattern outcome
            outcome = track_pattern_outcome(df, i, levels, pattern_type)
            
            patterns.append({
                'timestamp': candle['time'].isoformat(),
                'pattern_type': pattern_type,
                'swing_high': float(swing_high),
                'swing_low': float(swing_low),
                'entry_price': float(candle['close']),
                **{f'fib_{k}': float(v) for k,v in levels.items()},
                **outcome
            })
            
        logger.info(f"Identified {len(patterns)} valid patterns")
        return patterns
        
    except Exception as e:
        logger.error(f"Error analyzing patterns: {str(e)}")
        return []

def track_pattern_outcome(df: pd.DataFrame, entry_idx: int, levels: dict, pattern_type: str) -> dict:
    """Track the outcome of a pattern over subsequent candles"""
    entry_price = df.iloc[entry_idx]['close']
    outcome = {
        'sl_hit': False,
        'tp_hit': False,
        'exit_price': entry_price,
        'exit_time': df.iloc[entry_idx]['time'].isoformat(),
        'duration': 0,
        'profit': 0,
        'success': False
    }
    
    sl_level = levels['0.382'] if pattern_type == 'bullish' else levels['0.382']
    tp_level = levels['1.618']
    
    for j in range(1, min(50, len(df)-entry_idx-1)):
        future = df.iloc[entry_idx + j]
        outcome['duration'] = j
        outcome['exit_price'] = future['close']
        outcome['exit_time'] = future['time'].isoformat()
        
        if pattern_type == 'bullish':
            # Check for stop loss
            if future['low'] <= sl_level:
                outcome['sl_hit'] = True
                outcome['exit_price'] = sl_level
                break
            # Check for take profit
            elif future['high'] >= tp_level:
                outcome['tp_hit'] = True
                outcome['exit_price'] = tp_level
                outcome['success'] = True
                break
        else:
            # Bearish pattern checks
            if future['high'] >= sl_level:
                outcome['sl_hit'] = True
                outcome['exit_price'] = sl_level
                break
            elif future['low'] <= tp_level:
                outcome['tp_hit'] = True
                outcome['exit_price'] = tp_level
                outcome['success'] = True
                break
    
    # Calculate final profit
    if pattern_type == 'bullish':
        outcome['profit'] = outcome['exit_price'] - entry_price
    else:
        outcome['profit'] = entry_price - outcome['exit_price']
    
    return outcome

def update_learning_data(patterns: List[dict]) -> bool:
    """Store pattern analysis results in Firestore"""
    if db is None:
        logger.error("Cannot update learning data - Firestore not available")
        return False
    
    if not patterns:
        logger.info("No patterns to store")
        return True
        
    try:
        batch = db.batch()
        collection_ref = db.collection('learning_data')
        
        for pattern in patterns:
            doc_ref = collection_ref.document()
            batch.set(doc_ref, {
                **pattern,
                'processed_at': datetime.utcnow().isoformat(),
                'version': '1.0'
            })
        
        batch.commit()
        logger.info(f"Stored {len(patterns)} patterns in learning database")
        return True
    except Exception as e:
        logger.error(f"Error storing learning data: {str(e)}")
        return False

def update_model() -> bool:
    """Update trading model with new pattern data"""
    model_path = MODEL_DIR / 'fibonacci_model_current.pkl'
    
    try:
        # Load current model
        if not model_path.exists():
            logger.error("No current model found for updating")
            return False
            
        model = joblib.load(model_path)
        
        # Get recent patterns for training
        patterns = analyze_patterns(get_recent_market_data(48))
        if not patterns:
            logger.info("No new patterns found for model update")
            return False
            
        # Prepare training data
        X, y = prepare_training_data(patterns)
        if X is None or len(X) < 10:
            logger.warning(f"Insufficient training data ({len(X) if X else 0} samples)")
            return False
            
        # Update model
        model.fit(X, y)
        
        # Save new version
        version_name = f"fibonacci_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        version_path = MODEL_DIR / f"{version_name}.pkl"
        
        # Atomic save operation
        temp_path = version_path.with_suffix('.tmp')
        joblib.dump(model, temp_path)
        os.rename(temp_path, version_path)
        
        # Update current symlink
        temp_link = model_path.with_suffix('.tmp')
        os.symlink(version_path, temp_link)
        os.replace(temp_link, model_path)
        
        logger.info(f"Successfully updated model to version {version_name}")
        return True
        
    except Exception as e:
        logger.error(f"Model update failed: {str(e)}")
        
        # Clean up failed files
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        if 'version_path' in locals() and version_path.exists():
            version_path.unlink()
            
        return False

def prepare_training_data(patterns: List[dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert patterns to model training data"""
    if not patterns:
        return None, None
        
    features = []
    targets = []
    
    for pattern in patterns:
        try:
            # Feature vector
            feat = np.array([
                pattern['entry_price'] / pattern['swing_high'] - 1 if pattern['pattern_type'] == 'bullish' 
                    else pattern['entry_price'] / pattern['swing_low'] - 1,
                pattern['swing_high'] / pattern['swing_low'] - 1,
                1 if pattern['pattern_type'] == 'bullish' else 0,
                pattern['fib_0.382'] / pattern['entry_price'] - 1,
                pattern['fib_1.618'] / pattern['entry_price'] - 1,
                pattern['duration'] / 50.0,
                1 if pattern['success'] else 0
            ])
            
            # Target vector
            tgt = np.array([
                0.5,  # Default lot size
                1.0,  # Default SL multiplier
                1.5   # Default TP multiplier
            ])
            
            # Adjust targets based on outcome
            if pattern['success']:
                tgt[0] = min(0.6, tgt[0] * 1.1)  # Increase lot size slightly
                tgt[2] = min(2.0, tgt[2] * 1.1)  # Increase TP slightly
            else:
                tgt[0] = max(0.3, tgt[0] * 0.9)   # Decrease lot size
                tgt[1] = min(2.0, tgt[1] * 1.05)  # Increase SL slightly
                
            features.append(feat)
            targets.append(tgt)
            
        except Exception as e:
            logger.error(f"Error processing pattern for training: {str(e)}")
            continue
    
    if not features:
        return None, None
        
    return np.vstack(features), np.vstack(targets)

def main():
    """Main autonomous learning loop"""
    logger.info("Starting autonomous learning system")
    
    while True:
        try:
            # Collect and process data
            df = get_recent_market_data(48)
            if df.empty:
                logger.warning("No market data available - skipping cycle")
                time.sleep(3600)
                continue
                
            df = prepare_data(df)
            patterns = analyze_patterns(df)
            
            # Store results and update model
            if patterns:
                update_learning_data(patterns)
                
                # Update model every 12 hours
                if datetime.now().hour % 12 == 0:
                    logger.info("Starting scheduled model update")
                    update_model()
            
            logger.info("Completed learning cycle - sleeping for 1 hour")
            time.sleep(3600)
            
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {str(e)}")
            time.sleep(3600)

if __name__ == "__main__":
    main()