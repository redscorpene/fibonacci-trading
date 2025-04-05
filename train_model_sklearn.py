# train_model_sklearn.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import argparse
import matplotlib.pyplot as plt

def prepare_data(csv_file):
    """Prepare historical data for training"""
    # Load data
    df = pd.read_csv(csv_file)
    
    # Convert time column if it exists
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    # Calculate features
    df['candle_body'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['candle_range'] = df['high'] - df['low']
    
    # Identify breakout patterns
    for n in range(1, 6):
        df[f'breaks_candle_{n}_high'] = (df['close'] > df['high'].shift(n)).astype(int)
        df[f'breaks_candle_{n}_low'] = (df['close'] < df['low'].shift(n)).astype(int)
    
    df['breaks_pattern_bullish'] = ((df['breaks_candle_3_high'] == 1) | 
                                    (df['breaks_candle_4_high'] == 1) | 
                                    (df['breaks_candle_5_high'] == 1)).astype(int)
    
    df['breaks_pattern_bearish'] = ((df['breaks_candle_3_low'] == 1) | 
                                    (df['breaks_candle_4_low'] == 1) | 
                                    (df['breaks_candle_5_low'] == 1)).astype(int)
    
    return df

def generate_training_data(data, points_to_usd_factor=1.25/124.9, spread=3000):
    """Generate training data from historical patterns"""
    X = []
    y = []
    
    # Find all Fibonacci patterns
    for i in range(10, len(data) - 50):
        # Get candle
        candle = data.iloc[i]
        
        # Check for patterns
        bullish_break = any([candle[f'breaks_candle_{n}_high'] == 1 for n in range(3, 6)])
        bearish_break = any([candle[f'breaks_candle_{n}_low'] == 1 for n in range(3, 6)])
        
        if bullish_break or bearish_break:
            # Create features
            prev_candles = data.iloc[i-10:i]
            
            features = np.array([
                # Price action
                candle['close'] / candle['open'] - 1,
                candle['high'] / candle['low'] - 1,
                candle['candle_body'] / candle['candle_range'] if candle['candle_range'] != 0 else 0,
                
                # Recent volatility
                prev_candles['candle_range'].mean(),
                prev_candles['candle_range'].std(),
                
                # Pattern features
                1 if bullish_break else 0,
                1 if bearish_break else 0,
                
                # Volume
                candle['volume'] / prev_candles['volume'].mean() if 'volume' in candle.index and prev_candles['volume'].mean() > 0 else 1.0,
                
                # Time features
                np.sin(2 * np.pi * pd.to_datetime(candle['time']).hour / 24) if 'time' in candle.index else 0,
                
                # Account state
                1.0,  # Normalized balance
                0,    # No open trade
                0,    # No trade duration
                0,    # No fib redrawn
                
                # Progress toward doubling
                1.0,
                
                # Recent trend
                prev_candles['close'].iloc[-1] / prev_candles['close'].iloc[0] - 1,
                
                # Spread factor
                spread * points_to_usd_factor / 100
            ])
            
            # Find broken candle
            broken_idx = None
            for n in range(3, 6):
                if bullish_break and candle[f'breaks_candle_{n}_high'] == 1:
                    broken_idx = i - n
                    break
                elif bearish_break and candle[f'breaks_candle_{n}_low'] == 1:
                    broken_idx = i - n
                    break
            
            # Calculate Fibonacci levels
            if broken_idx is not None:
                if bullish_break:
                    fib_high = candle['close']
                    fib_low = data.iloc[broken_idx]['low']
                else:
                    fib_high = data.iloc[broken_idx]['high']
                    fib_low = candle['close']
                    
                fib_range = fib_high - fib_low
                
                # For simulation, use some default values for risk parameters
                lot_pct = 0.5  # Default 50% of max risk
                sl_mult = 1.0  # Default stop loss at 0.0 Fib level
                tp_mult = 1.0  # Default take profit at 161.8 Fib level
                
                # Add to training data
                X.append(features)
                y.append([lot_pct, sl_mult, tp_mult])
    
    return np.array(X), np.array(y)

def train_model(data_path, output_dir="app/models"):
    """Train the Fibonacci trading model"""
    # Load and prepare data
    print("Loading data...")
    data = prepare_data(data_path)
    
    # Generate training data
    print("Generating training data...")
    X, y = generate_training_data(data)
    
    if len(X) == 0:
        print("No patterns found in data for training. Please check your data.")
        return None, None
    
    print(f"Found {len(X)} training examples")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a model for each output dimension
    print("Creating and training models...")
    models = []
    
    for i in range(3):
        # Create a GradientBoostingRegressor for this output
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Train the model
        model.fit(X, y[:, i])
        models.append(model)
        
        print(f"Trained model for output {i+1}/3")
    
    # Save models
    model_path = os.path.join(output_dir, "fibonacci_model.pkl")
    joblib.dump(models, model_path)
    print(f"Models saved to {model_path}")
    
    # Create and save scaler
    scaler = StandardScaler()
    scaler.fit(X)
    scaler_path = os.path.join(output_dir, "state_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    feature_names = [
        'Candle Return', 'Candle Range', 'Body Ratio',
        'Recent Volatility', 'Volatility Std',
        'Bullish Pattern', 'Bearish Pattern',
        'Volume Ratio', 'Hour of Day',
        'Balance Ratio', 'Trade Open', 'Trade Duration', 'Fib Redrawn',
        'Progress to Double', 'Recent Trend', 'Spread Factor'
    ]
    
    for i, model in enumerate(models):
        plt.subplot(1, 3, i+1)
        plt.bar(range(len(feature_names)), model.feature_importances_)
        plt.xticks(range(len(feature_names)), feature_names, rotation=90)
        plt.title(f"Feature Importance for {'Lot Size' if i==0 else 'SL Mult' if i==1 else 'TP Mult'}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    
    return models, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Fibonacci trading model')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file with candle data')
    parser.add_argument('--output', type=str, default='app/models', help='Output directory for model files')
    
    args = parser.parse_args()
    
    train_model(args.data, args.output)