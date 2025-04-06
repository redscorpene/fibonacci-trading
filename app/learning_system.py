import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import json
import logging
import os
from sqlalchemy.orm import Session
from .database import Trade

class ContinuousLearningSystem:
    def __init__(self, model_path="app/models/fibonacci_model.pkl", learning_rate=0.0001):
        """Initialize the continuous learning system"""
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.logger = logging.getLogger(__name__)
        
        # Load model if it exists, otherwise create a placeholder
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                self.model = None
        else:
            self.logger.warning(f"Model not found at {model_path}, waiting for initial model")
            self.model = None
            
        # Check if we should use autonomous learning
        self.use_autonomous_learning = os.environ.get('USE_AUTONOMOUS_LEARNING', '0') == '1'
        if self.use_autonomous_learning:
            self.logger.info("Autonomous learning is enabled")
            try:
                from google.cloud import firestore
                self.db_firestore = firestore.Client()
                self.firestore_available = True
                self.logger.info("Connected to Firestore for autonomous learning")
            except Exception as e:
                self.logger.error(f"Could not connect to Firestore: {e}")
                self.firestore_available = False
    
    def get_completed_trades(self, db: Session, since=None):
        """Retrieve completed trades from database"""
        if since is None:
            since = datetime.utcnow() - timedelta(days=3)
            
        try:
            # Query database for completed trades
            trades = db.query(Trade).filter(
                Trade.trade_status == "completed",
                Trade.completion_time >= since
            ).all()
            
            self.logger.info(f"Retrieved {len(trades)} completed trades for learning")
            return trades
        
        except Exception as e:
            self.logger.error(f"Error retrieving completed trades: {e}")
            return []
    
    def get_autonomous_learning_data(self, days=7):
        """Get learning data collected autonomously from Firestore"""
        if not self.use_autonomous_learning or not self.firestore_available:
            self.logger.info("Autonomous learning is disabled or Firestore unavailable")
            return []
            
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            # Query Firestore for autonomous learning data
            docs = self.db_firestore.collection('learning_data').where(
                'timestamp', '>=', cutoff_time.isoformat()
            ).stream()
            
            # Process data for learning
            patterns = []
            for doc in docs:
                data = doc.to_dict()
                
                # Basic validation
                if 'pattern_type' not in data or 'profit' not in data:
                    continue
                    
                patterns.append(data)
            
            self.logger.info(f"Retrieved {len(patterns)} autonomous learning patterns")
            return patterns
        except Exception as e:
            self.logger.error(f"Error retrieving autonomous learning data: {e}")
            return []
    
    def prepare_learning_data(self, trades):
        """Prepare training data from completed trades"""
        if not trades:
            self.logger.info("No completed trades to learn from")
            return None, None
        
        features = []
        targets = []
        
        for trade in trades:
            try:
                # Extract original features used for prediction
                if not trade.original_features:
                    continue
                
                original_features = np.array(json.loads(trade.original_features))
                if len(original_features) == 0:
                    continue
                
                # Original model output (action)
                if not trade.original_action:
                    continue
                
                original_action = np.array(json.loads(trade.original_action))
                if len(original_action) == 0:
                    continue
                
                # Get trade result
                profit = trade.profit or 0
                expected_profit = trade.expected_profit or 0
                
                # Calculate reward score (-1 to 1)
                if expected_profit > 0:
                    # If profitable, score based on how much of expected profit was achieved
                    reward_score = min(profit / expected_profit, 1.0) if expected_profit != 0 else 0
                else:
                    # If loss, score based on how much loss was minimized
                    reward_score = max(profit / expected_profit, -1.0) if expected_profit != 0 else 0
                
                # Adjust original action based on reward
                adjusted_action = original_action.copy()
                
                # Adjust lot size percentage based on profit/loss
                if reward_score > 0:
                    # If profitable, can slightly increase risk
                    adjusted_action[0] = min(original_action[0] * (1 + 0.1 * reward_score), 1.0)
                else:
                    # If loss, reduce risk
                    adjusted_action[0] = max(original_action[0] * (1 + 0.2 * reward_score), 0.1)
                
                # Adjust SL/TP multipliers based on how close actual price came to them
                sl_improvement = trade.sl_improvement or 0
                tp_improvement = trade.tp_improvement or 0
                
                if sl_improvement != 0:
                    adjusted_action[1] = max(min(original_action[1] * (1 + 0.1 * sl_improvement), 3.0), 0.5)
                
                if tp_improvement != 0:
                    adjusted_action[2] = max(min(original_action[2] * (1 + 0.1 * tp_improvement), 3.0), 0.5)
                
                # Add to training data
                features.append(original_features)
                targets.append(adjusted_action)
                
            except Exception as e:
                self.logger.error(f"Error processing trade for learning: {e}")
                continue
        
        if not features:
            return None, None
            
        return np.array(features), np.array(targets)
    
    def prepare_autonomous_learning_data(self, patterns):
        """Prepare training data from autonomous learning patterns"""
        if not patterns:
            return None, None
            
        features = []
        targets = []
        
        for pattern in patterns:
            try:
                # Create feature vector similar to what's used in real-time prediction
                # This is a simplified example - you would need to match your feature extraction
                pattern_type = pattern.get('pattern_type')
                is_bullish = 1 if pattern_type == 'bullish' else 0
                is_bearish = 1 if pattern_type == 'bearish' else 0
                
                # Basic features
                feature = np.array([
                    0.0,  # Price action (placeholder)
                    0.0,  # Candle range (placeholder)
                    0.0,  # Body ratio (placeholder)
                    0.0,  # Volatility (placeholder)
                    0.0,  # Volatility std (placeholder)
                    is_bullish,  # Pattern type bullish
                    is_bearish,  # Pattern type bearish
                    0.0,  # Volume (placeholder)
                    0.0,  # Time (placeholder)
                    1.0,  # Account state (normalized)
                    0.0,  # No open trade
                    0.0,  # No trade duration
                    0.0,  # No fib redrawn
                    1.0,  # Target metric
                    0.0,  # Market trend (placeholder)
                    0.0   # Spread factor (placeholder)
                ])
                
                # Calculate success metrics
                success = pattern.get('tp_hit', False)
                profit = pattern.get('profit', 0)
                fib_range = pattern.get('fib_range', 0)
                
                # Default action values
                lot_pct = 0.5  # Default
                sl_mult = 1.0  # Default
                tp_mult = 1.0  # Default
                
                # Adjust based on success/failure
                if success:
                    lot_pct = min(0.6, lot_pct * 1.1)  # Slightly increase lot size for successful patterns
                else:
                    lot_pct = max(0.3, lot_pct * 0.9)  # Slightly decrease lot size for unsuccessful patterns
                
                # Calculate reward score based on profit relative to range
                if fib_range > 0:
                    reward_score = min(max(profit / fib_range, -1.0), 1.0)
                    
                    # Adjust SL/TP multipliers based on outcome
                    sl_mult = max(min(sl_mult * (1 + 0.1 * reward_score), 2.0), 0.5)
                    tp_mult = max(min(tp_mult * (1 + 0.1 * reward_score), 2.0), 0.5)
                
                # Add to training data
                features.append(feature)
                targets.append([lot_pct, sl_mult, tp_mult])
                
            except Exception as e:
                self.logger.error(f"Error processing autonomous pattern for learning: {e}")
                continue
        
        if not features:
            return None, None
            
        return np.array(features), np.array(targets)
    
    def update_model(self, db: Session):
        """Update the model based on recent trading results and autonomous learning"""
        try:
            # Check if model is loaded
            if self.model is None:
                self.logger.error("No model loaded, cannot update")
                return False
            
            # Get recent completed trades
            trades = self.get_completed_trades(db)
            
            # Prepare data from trades
            X_trades, y_trades = self.prepare_learning_data(trades)
            
            # Get autonomous learning data if enabled
            if self.use_autonomous_learning and self.firestore_available:
                patterns = self.get_autonomous_learning_data()
                X_patterns, y_patterns = self.prepare_autonomous_learning_data(patterns)
            else:
                X_patterns, y_patterns = None, None
            
            # Combine data sources if both available
            if X_trades is not None and X_patterns is not None:
                X = np.vstack([X_trades, X_patterns])
                y = np.vstack([y_trades, y_patterns])
                self.logger.info(f"Combined {len(X_trades)} trade examples with {len(X_patterns)} autonomous examples")
            elif X_trades is not None:
                X, y = X_trades, y_trades
                self.logger.info(f"Using {len(X_trades)} trade examples")
            elif X_patterns is not None:
                X, y = X_patterns, y_patterns
                self.logger.info(f"Using {len(X_patterns)} autonomous examples")
            else:
                self.logger.info("No learning data available from any source")
                return False
            
            if len(X) == 0:
                self.logger.info("No valid learning data could be extracted")
                return False
            
            self.logger.info(f"Training model on {len(X)} examples")
            
            # Update each model in the ensemble
            for i, model in enumerate(self.model):
                # Train on the data
                model.fit(X, y[:, i])
            
            # Save updated model
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = f"app/models/fibonacci_model_backup_{timestamp}.pkl"
            
            # Create a backup
            joblib.dump(self.model, backup_path)
            self.logger.info(f"Model backup saved to {backup_path}")
                
            # Replace current model
            joblib.dump(self.model, self.model_path)
            
            self.logger.info("Model updated and saved")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating model: {e}")
            return False