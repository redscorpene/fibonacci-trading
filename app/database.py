from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime
import json

# Check if running in cloud environment
is_cloud_environment = os.environ.get('K_SERVICE') is not None or os.environ.get('GOOGLE_CLOUD_PROJECT') is not None

# Initialize Firestore conditionally
if is_cloud_environment:
    try:
        from google.cloud import firestore
        db_firestore = firestore.Client()
        use_firestore = True
    except (ImportError, Exception):
        use_firestore = False
else:
    use_firestore = False

# SQLite setup (for local development or fallback)
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./data/trading.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Trade(Base):
    __tablename__ = "trades"

    trade_id = Column(String, primary_key=True)
    timestamp = Column(String, primary_key=True)
    action = Column(String)
    lot_size = Column(Float)
    entry_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    confidence = Column(Float)
    message = Column(String)
    redraw_fibo = Column(Boolean, default=False)
    original_features = Column(Text, nullable=True)  # JSON string
    original_action = Column(Text, nullable=True)    # JSON string
    expected_profit = Column(Float, nullable=True)
    profit = Column(Float, nullable=True)
    sl_improvement = Column(Float, nullable=True)
    tp_improvement = Column(Float, nullable=True)
    trade_status = Column(String, default="opened")
    creation_time = Column(DateTime, default=datetime.utcnow)
    completion_time = Column(DateTime, nullable=True)

# Function to handle database operations with appropriate backend
def store_trade(trade_data):
    if use_firestore:
        try:
            # Store in Firestore
            trades_ref = db_firestore.collection('trades')
            doc_ref = trades_ref.document(f"{trade_data['trade_id']}_{trade_data['timestamp']}")
            doc_ref.set(trade_data)
            return True
        except Exception as e:
            print(f"Error storing in Firestore: {e}")
            # Fall back to SQLite
            store_trade_sqlite(trade_data)
    else:
        # Use SQLite
        return store_trade_sqlite(trade_data)

def store_trade_sqlite(trade_data):
    try:
        db = SessionLocal()
        trade = Trade(**trade_data)
        db.add(trade)
        db.commit()
        return True
    except Exception as e:
        print(f"Error storing in SQLite: {e}")
        if 'db' in locals():
            db.rollback()
        return False
    finally:
        if 'db' in locals():
            db.close()

# Create tables (SQLite only)
def create_db_and_tables():
    if not use_firestore:
        Base.metadata.create_all(bind=engine)