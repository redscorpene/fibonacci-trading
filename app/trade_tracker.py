import json
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from .database import Trade

logger = logging.getLogger(__name__)

class TradeTracker:
    def __init__(self):
        """Initialize the trade tracker"""
        self.logger = logging.getLogger(__name__)
    
    def record_trade_start(self, db: Session, trade_id, signal, features, action, expected_profit):
        """Record the start of a trade"""
        try:
            # Create a new trade record
            new_trade = Trade(
                trade_id=trade_id,
                timestamp=signal['timestamp'],
                action=signal['action'],
                lot_size=signal['lot_size'],
                entry_price=signal.get('entry_price'),
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit'),
                confidence=signal['confidence'],
                message=signal['message'],
                redraw_fibo=signal.get('redraw_fibo', False),
                original_features=json.dumps(features.tolist()),
                original_action=json.dumps(action.tolist()),
                expected_profit=expected_profit,
                trade_status='opened',
                creation_time=datetime.utcnow()
            )
            
            # Add to database
            db.add(new_trade)
            db.commit()
            
            self.logger.info(f"Recorded trade start: {trade_id}")
            return True
        
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error recording trade start: {e}")
            return False
    
    def record_trade_completion(self, db: Session, trade_id, timestamp, profit, sl_improvement=0, tp_improvement=0):
        """Record the completion of a trade"""
        try:
            # Find the trade
            trade = db.query(Trade).filter(Trade.trade_id == trade_id, Trade.timestamp == timestamp).first()
            
            if not trade:
                self.logger.error(f"Trade not found: {trade_id}, {timestamp}")
                return False
            
            # Update the trade record
            trade.profit = profit
            trade.sl_improvement = sl_improvement
            trade.tp_improvement = tp_improvement
            trade.trade_status = 'completed'
            trade.completion_time = datetime.utcnow()
            
            # Commit changes
            db.commit()
            
            self.logger.info(f"Recorded trade completion: {trade_id}, Profit: {profit}")
            return True
        
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error recording trade completion: {e}")
            return False