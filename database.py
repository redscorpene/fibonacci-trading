from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(36), nullable=False, unique=True)
    trade_status = Column(String(20), nullable=False, index=True)
    trade_type = Column(String(10))  # 'BUY' or 'SELL'
    original_features = Column(Text)  # JSON string of original features
    original_action = Column(Text)    # JSON string of original action
    profit = Column(Float)
    expected_profit = Column(Float)
    sl_improvement = Column(Float)
    tp_improvement = Column(Float)
    completion_time = Column(DateTime, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    fib_redrawn = Column(Boolean, default=False)
    duration_minutes = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)
    lot_size = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    
    def __repr__(self):
        return f"<Trade(id={self.id}, status={self.trade_status}, profit={self.profit:.2f})>"

class ModelVersion(Base):
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True)
    version_name = Column(String(64), unique=True)
    path = Column(String(256))
    created_at = Column(DateTime, default=datetime.utcnow)
    performance_metrics = Column(Text)  # JSON string
    is_active = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<ModelVersion(name={self.version_name}, active={self.is_active})>"