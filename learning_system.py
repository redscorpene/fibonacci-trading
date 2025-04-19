import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from .database import Trade, ModelVersion

class ContinuousLearningSystem:
    def __init__(self, model_path: Optional[str] = None, learning_rate: float = 0.0001):
        """Initialize the continuous learning system with proper configuration"""
        self.logger = self._configure_logging()
        self.learning_rate = learning_rate
        self.model, self.model_path = self._initialize_model(model_path)
        self._init_autonomous_learning()
    
    def _configure_logging(self) -> logging.Logger:
        """Set up standardized logging"""
        logger = logging.getLogger('fibonacci_learning')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def _initialize_model(self, model_path: Optional[str]) -> Tuple[Optional[object], Path]:
        """Initialize model with proper path handling"""
        # Determine model directory
        model_dir = Path(os.getenv('MODEL_DIR', '/app/models'))
        if not model_dir.exists():
            model_dir = Path(__file__).parent.parent / 'models'
            model_dir.mkdir(exist_ok=True)
        
        # Resolve model path
        resolved_path = Path(model_path) if model_path else model_dir / 'fibonacci_model_current.pkl'
        
        # Load model with error handling
        try:
            if resolved_path.exists():
                model = joblib.load(resolved_path)
                self.logger.info(f"Successfully loaded model from {resolved_path}")
                return model, resolved_path
            else:
                self.logger.warning(f"No model found at {resolved_path}")
                return None, resolved_path
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return None, resolved_path

    def _init_autonomous_learning(self):
        """Initialize autonomous learning components"""
        self.use_autonomous_learning = os.getenv('USE_AUTONOMOUS_LEARNING', '0') == '1'
        self.firestore_available = False
        
        if self.use_autonomous_learning:
            try:
                from google.cloud import firestore
                self.db_firestore = firestore.Client()
                self.firestore_available = True
                self.logger.info("Firestore client initialized for autonomous learning")
            except Exception as e:
                self.logger.error(f"Failed to initialize Firestore: {str(e)}")

    def get_completed_trades(self, db: Session, since: Optional[datetime] = None) -> list:
        """Retrieve completed trades with proper type hints and error handling"""
        if since is None:
            since = datetime.utcnow() - timedelta(days=3)
        
        try:
            trades = db.query(Trade).filter(
                Trade.trade_status == "completed",
                Trade.completion_time >= since
            ).order_by(Trade.completion_time.desc()).all()
            
            self.logger.info(f"Retrieved {len(trades)} completed trades since {since}")
            return trades
        except Exception as e:
            self.logger.error(f"Error retrieving trades: {str(e)}")
            return []

    def prepare_learning_data(self, trades: list) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data with comprehensive validation"""
        if not trades:
            self.logger.info("No trades provided for learning")
            return None, None
        
        features = []
        targets = []
        
        for trade in trades:
            try:
                # Validate and parse original features
                if not trade.original_features:
                    continue
                    
                original_features = np.array(json.loads(trade.original_features))
                if original_features.size == 0:
                    continue
                
                # Validate and parse original action
                if not trade.original_action:
                    continue
                    
                original_action = np.array(json.loads(trade.original_action))
                if original_action.size == 0:
                    continue
                
                # Calculate reward metrics
                profit = trade.profit if trade.profit is not None else 0
                expected_profit = trade.expected_profit if trade.expected_profit is not None else 0
                
                # Enhanced reward calculation
                if expected_profit != 0:
                    reward_score = np.clip(profit / expected_profit, -1.0, 1.0)
                else:
                    reward_score = 0
                
                # Create adjusted action
                adjusted_action = original_action.copy()
                
                # Dynamic lot size adjustment
                adjusted_action[0] = np.clip(
                    original_action[0] * (1 + 0.15 * reward_score),
                    0.1,  # Minimum lot size
                    1.0   # Maximum lot size
                )
                
                # SL/TP adjustment based on performance
                sl_improvement = trade.sl_improvement or 0
                tp_improvement = trade.tp_improvement or 0
                
                adjusted_action[1] = np.clip(
                    original_action[1] * (1 + 0.1 * sl_improvement),
                    0.5,  # Min SL multiplier
                    3.0   # Max SL multiplier
                )
                
                adjusted_action[2] = np.clip(
                    original_action[2] * (1 + 0.1 * tp_improvement),
                    0.5,  # Min TP multiplier
                    3.0   # Max TP multiplier
                )
                
                features.append(original_features)
                targets.append(adjusted_action)
                
            except Exception as e:
                self.logger.error(f"Error processing trade {trade.id}: {str(e)}")
                continue
        
        if not features:
            return None, None
            
        return np.vstack(features), np.vstack(targets)

    def update_model(self, db: Session) -> bool:
        """Comprehensive model update with version control"""
        if self.model is None:
            self.logger.error("Cannot update - no model loaded")
            return False
            
        try:
            # Get learning data from multiple sources
            trade_data = self.get_completed_trades(db)
            X_trades, y_trades = self.prepare_learning_data(trade_data)
            
            # Get autonomous learning data if enabled
            if self.use_autonomous_learning and self.firestore_available:
                patterns = self._get_autonomous_learning_data()
                X_patterns, y_patterns = self._prepare_autonomous_data(patterns)
            else:
                X_patterns, y_patterns = None, None
            
            # Combine data sources
            X, y = self._combine_data_sources(X_trades, y_trades, X_patterns, y_patterns)
            if X is None:
                return False
                
            # Update model
            self._train_model(X, y)
            
            # Save new version
            version_name = f"fibonacci_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            return self._save_model_version(version_name, db)
            
        except Exception as e:
            self.logger.error(f"Model update failed: {str(e)}")
            return False

    def _get_autonomous_learning_data(self, days: int = 7) -> list:
        """Retrieve autonomous learning data from Firestore"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            docs = self.db_firestore.collection('learning_data').where(
                'timestamp', '>=', cutoff_time.isoformat()
            ).stream()
            
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            self.logger.error(f"Error getting autonomous data: {str(e)}")
            return []

    def _prepare_autonomous_data(self, patterns: list) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare autonomous learning patterns for training"""
        if not patterns:
            return None, None
            
        features = []
        targets = []
        
        for pattern in patterns:
            try:
                # Feature extraction logic
                feature = self._extract_pattern_features(pattern)
                target = self._calculate_pattern_target(pattern)
                
                features.append(feature)
                targets.append(target)
            except Exception as e:
                self.logger.error(f"Error processing pattern: {str(e)}")
                continue
                
        if not features:
            return None, None
            
        return np.vstack(features), np.vstack(targets)

    def _extract_pattern_features(self, pattern: dict) -> np.ndarray:
        """Extract features from a pattern dictionary"""
        # Implement your feature extraction logic here
        return np.zeros(16)  # Placeholder - replace with actual features

    def _calculate_pattern_target(self, pattern: dict) -> np.ndarray:
        """Calculate target values from pattern results"""
        # Implement your target calculation logic here
        return np.zeros(3)  # Placeholder - replace with actual targets

    def _combine_data_sources(self, X_trades, y_trades, X_patterns, y_patterns):
        """Combine data from different sources with validation"""
        if X_trades is not None and X_patterns is not None:
            try:
                X = np.vstack([X_trades, X_patterns])
                y = np.vstack([y_trades, y_patterns])
                self.logger.info(f"Combined {len(X_trades)} trade and {len(X_patterns)} pattern examples")
                return X, y
            except ValueError as e:
                self.logger.error(f"Data shape mismatch: {str(e)}")
                return None, None
        elif X_trades is not None:
            return X_trades, y_trades
        elif X_patterns is not None:
            return X_patterns, y_patterns
        else:
            self.logger.warning("No valid training data available")
            return None, None

    def _train_model(self, X: np.ndarray, y: np.ndarray):
        """Train model with new data"""
        try:
            for i, model in enumerate(self.model):
                model.partial_fit(X, y[:, i])
            self.logger.info(f"Model updated with {len(X)} new examples")
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

    def _save_model_version(self, version_name: str, db: Session) -> bool:
        """Save new model version with database tracking"""
        try:
            # Create versioned file path
            version_path = self.model_path.parent / f"{version_name}.pkl"
            
            # Save model
            joblib.dump(self.model, version_path)
            
            # Update symlink atomically
            temp_path = self.model_path.with_suffix('.tmp')
            os.symlink(version_path, temp_path)
            os.replace(temp_path, self.model_path)
            
            # Record in database
            model_version = ModelVersion(
                version_name=version_name,
                path=str(version_path),
                is_active=True
            )
            db.add(model_version)
            db.commit()
            
            self.logger.info(f"Saved new model version: {version_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model version: {str(e)}")
            db.rollback()
            
            # Clean up failed files
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
            if 'version_path' in locals() and version_path.exists():
                version_path.unlink()
                
            return False