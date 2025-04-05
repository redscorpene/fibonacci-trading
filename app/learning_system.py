import numpy as np
import json
import logging
import os
from datetime import datetime, timedelta
import joblib

from sqlalchemy.orm import Session
from .database import Trade

from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class ContinuousLearningSystem:
    def __init__(self, model_path="app/models/fibonacci_model.joblib"):
        """Initialize the continuous learning system"""
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
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
    
    def save_model_to_gcs(self, local_path, bucket_name="fibonacci-trading-models"):
        """Save model to Google Cloud Storage"""
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            
            # Get filename from path
            filename = os.path.basename(local_path)
            
            # Upload to GCS
            blob = bucket.blob(filename)
            blob.upload_from_filename(local_path)
            
            self.logger.info(f"Model saved to GCS: gs://{bucket_name}/{filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model to GCS: {e}")
            return False
    
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
    
    def update_model(self, db: Session):
        """Update the model based on recent trading results"""
        try:
            # Get recent completed trades
            trades = self.get_completed_trades(db)
            
            if not trades:
                self.logger.info("No trades to learn from")
                return False
            
            self.logger.info(f"Found {len(trades)} completed trades for learning")
            
            # Prepare training data
            X, y = self.prepare_learning_data(trades)
            
            if X is None or len(X) == 0:
                self.logger.info("No valid learning data could be extracted from trades")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # If model doesn't exist, create a new one
            if self.model is None:
                self.model = RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42, 
                    max_depth=10, 
                    min_samples_split=5
                )
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Fit the model
            self.model.fit(X_train, y_train)
            
            # Validate the model
            y_pred = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_pred)
            self.logger.info(f"Model validation MSE: {val_mse}")
            
            # Save updated model
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = f"app/models/fibonacci_model_backup_{timestamp}.joblib"
            
            # Create a backup
            joblib.dump(self.model, backup_path)
            self.logger.info(f"Model backup saved to {backup_path}")
            
            # Save to GCS if running in cloud environment
            if os.environ.get("GOOGLE_CLOUD_PROJECT"):
                self.save_model_to_gcs(backup_path)
                
            # Replace current model
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.model_path.replace('.joblib', '_scaler.joblib'))
            
            self.logger.info("Model updated and saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating model: {e}")
            return False

    def predict(self, features):
        """Make predictions using the trained model"""
        try:
            if self.model is None:
                self.logger.error("No model loaded for prediction")
                return None
            
            # Scale the input features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)
            
            return prediction
        
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return None