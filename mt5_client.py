import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import time
import logging
import configparser
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingClient")

class MT5JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MT5 data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.uint64, np.uint32, np.uint16, np.uint8)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '_asdict'):  # For named tuples
            return {k: self.default(v) for k, v in obj._asdict().items()}
        return super().default(obj)

class FibonacciTradingClient:
    def __init__(self, config_file='config.ini'):
        """Initialize the Fibonacci trading client"""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        # API settings
        self.api_endpoint = self.config.get('API', 'endpoint')
        self.api_key = self.config.get('API', 'key')
        
        # Trading settings
        self.symbol = self.config.get('Trading', 'symbol', fallback='BTCUSD')
        self.timeframe = self.get_timeframe(self.config.get('Trading', 'timeframe', fallback='5m'))
        self.candles_lookback = self.config.getint('Trading', 'candles_lookback', fallback=50)
        self.check_interval = self.config.getint('Trading', 'check_interval', fallback=5)
        
        # State variables
        self.running = False
        self.active_positions: Dict[int, Dict] = {}
        self.last_analysis_time = None
    
    def get_timeframe(self, timeframe_str: str) -> int:
        """Convert timeframe string to MT5 timeframe constant"""
        timeframes = {
            '1m': mt5.TIMEFRAME_M1,
            '5m': mt5.TIMEFRAME_M5,
            '15m': mt5.TIMEFRAME_M15,
            '30m': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1
        }
        return timeframes.get(timeframe_str.lower(), mt5.TIMEFRAME_M5)
    
    def initialize(self) -> bool:
        """Initialize connection to MT5"""
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False
        
        logger.info(f"MT5 connected. Terminal info: {mt5.terminal_info()}")
        logger.info(f"Logged in as: {mt5.account_info().name}, Balance: ${mt5.account_info().balance}")
        
        # Check if symbol is available
        if not mt5.symbol_select(self.symbol, True):
            logger.error(f"Symbol {self.symbol} not found in MT5")
            return False
        
        logger.info(f"Trading {self.symbol} on {self.get_timeframe_name(self.timeframe)}")
        return True
    
    def get_timeframe_name(self, timeframe: int) -> str:
        """Get timeframe name from MT5 timeframe constant"""
        timeframes = {
            mt5.TIMEFRAME_M1: "1-minute",
            mt5.TIMEFRAME_M5: "5-minute",
            mt5.TIMEFRAME_M15: "15-minute",
            mt5.TIMEFRAME_M30: "30-minute",
            mt5.TIMEFRAME_H1: "1-hour",
            mt5.TIMEFRAME_H4: "4-hour",
            mt5.TIMEFRAME_D1: "Daily"
        }
        return timeframes.get(timeframe, "Unknown")
    
    def get_market_data(self) -> Optional[Dict]:
        """Get current market data from MT5"""
        try:
            # Get current tick data
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                logger.error("Failed to get current tick data")
                return None
            
            # Get candles for pattern analysis
            candles = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.candles_lookback)
            if candles is None or len(candles) == 0:
                logger.error("Failed to get candle data")
                return None
            
            # Convert candles to serializable format
            candles_data = []
            for candle in candles:
                candles_data.append({
                    "time": datetime.fromtimestamp(candle['time']).isoformat(),
                    "open": float(candle['open']),
                    "high": float(candle['high']),
                    "low": float(candle['low']),
                    "close": float(candle['close']),
                    "volume": float(candle['tick_volume'])
                })
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return None
            
            account_data = {
                "balance": float(account_info.balance),
                "equity": float(account_info.equity),
                "margin": float(account_info.margin),
                "free_margin": float(account_info.margin_free)
            }
            
            # Get active positions
            positions = mt5.positions_get(symbol=self.symbol)
            active_positions = []
            
            if positions:
                for position in positions:
                    if position.magic == 234000:  # Our system's magic number
                        pos_data = {
                            "ticket": int(position.ticket),
                            "type": "buy" if position.type == 0 else "sell",
                            "volume": float(position.volume),
                            "open_price": float(position.price_open),
                            "sl": float(position.sl),
                            "tp": float(position.tp),
                            "profit": float(position.profit),
                            "fib_redrawn": position.ticket in self.active_positions and 
                                          self.active_positions[position.ticket].get("fib_redrawn", False)
                        }
                        
                        if position.ticket in self.active_positions:
                            pos_data.update({
                                "broken_candle_high": self.active_positions[position.ticket].get("broken_candle_high"),
                                "broken_candle_low": self.active_positions[position.ticket].get("broken_candle_low"),
                                "fib_0": self.active_positions[position.ticket].get("fib_0"),
                                "trade_id": self.active_positions[position.ticket].get("trade_id"),
                                "timestamp": self.active_positions[position.ticket].get("timestamp")
                            })
                        
                        active_positions.append(pos_data)
            
            return {
                "candles": candles_data,
                "account": account_data,
                "active_positions": active_positions,
                "current_tick": {
                    "bid": float(tick.bid),
                    "ask": float(tick.ask),
                    "time": datetime.fromtimestamp(tick.time).isoformat(),
                    "spread": float(tick.ask - tick.bid)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}", exc_info=True)
            return None
    
    def analyze_market(self) -> Optional[Dict]:
        """Get market data and send to AI for analysis"""
        market_data = self.get_market_data()
        if market_data is None:
            return None
        
        request_data = {
            "api_key": self.api_key,
            "candles": market_data["candles"],
            "current_tick": market_data["current_tick"],
            "account": market_data["account"],
            "active_positions": market_data["active_positions"]
        }
        
        try:
            response = requests.post(
                f"{self.api_endpoint}/analyze",
                json=json.dumps(request_data, cls=MT5JSONEncoder),
                headers={'Content-Type': 'application/json'},
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending data to API: {str(e)}", exc_info=True)
            return None
    
    def execute_trade_signal(self, signal: Dict) -> bool:
        """Execute the trade signal from AI in MT5"""
        try:
            if signal["action"] == "HOLD":
                logger.info(f"Signal: HOLD - {signal['message']}")
                return True
            
            # Prepare common request fields
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": float(signal["lot_size"]),
                "sl": float(signal["stop_loss"]),
                "tp": float(signal["take_profit"]),
                "deviation": 10,
                "magic": 234000,
                "comment": f"AI Fib Strategy {'Redraw' if signal.get('redraw_fibo', False) else ''} "
                          f"trade_id:{signal.get('trade_id', '')} "
                          f"timestamp:{datetime.now().isoformat()}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }
            
            # Set order type specific fields
            if signal["action"] == "BUY":
                request.update({
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": mt5.symbol_info_tick(self.symbol).ask
                })
            elif signal["action"] == "SELL":
                request.update({
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": mt5.symbol_info_tick(self.symbol).bid
                })
            
            # Execute trade
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Trade executed: {signal['action']} {result.order} @ {result.price}")
                
                # Store position metadata
                self.active_positions[result.order] = {
                    "type": signal["action"].lower(),
                    "volume": float(signal["lot_size"]),
                    "open_price": float(signal["entry_price"]),
                    "sl": float(signal["stop_loss"]),
                    "tp": float(signal["take_profit"]),
                    "fib_redrawn": signal.get("redraw_fibo", False),
                    "trade_id": signal.get("trade_id", ""),
                    "timestamp": datetime.now().isoformat()
                }
                return True
            else:
                logger.error(f"Trade failed: {result.comment}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}", exc_info=True)
            return False
    
    def monitor_trades(self):
        """Monitor active trades and report results"""
        while self.running:
            try:
                positions = mt5.positions_get(symbol=self.symbol)
                current_tickets = set()
                
                if positions:
                    for position in positions:
                        if position.magic == 234000:
                            ticket = position.ticket
                            current_tickets.add(ticket)
                            
                            if ticket not in self.active_positions:
                                # Parse metadata from comment
                                comment = position.comment
                                metadata = {
                                    "trade_id": "",
                                    "timestamp": ""
                                }
                                
                                if "trade_id:" in comment:
                                    metadata["trade_id"] = comment.split("trade_id:")[1].split()[0]
                                if "timestamp:" in comment:
                                    metadata["timestamp"] = comment.split("timestamp:")[1].split()[0]
                                
                                self.active_positions[ticket] = {
                                    "type": "buy" if position.type == 0 else "sell",
                                    "volume": float(position.volume),
                                    "open_price": float(position.price_open),
                                    "sl": float(position.sl),
                                    "tp": float(position.tp),
                                    "fib_redrawn": False,
                                    **metadata
                                }
                
                # Check for closed positions
                for ticket in set(self.active_positions.keys()) - current_tickets:
                    position_data = self.active_positions[ticket]
                    
                    # Get closing deal details
                    history = mt5.history_deals_get(
                        datetime.now() - timedelta(days=1),
                        datetime.now(),
                        group=position_data.get("trade_id", "")
                    )
                    
                    if history:
                        closing_deal = next((d for d in history if d.entry == 1), None)
                        if closing_deal:
                            profit = float(closing_deal.profit)
                            close_price = float(closing_deal.price)
                            
                            # Calculate performance metrics
                            sl_distance, tp_distance = 0, 0
                            if position_data["type"] == "buy":
                                if close_price <= position_data["sl"]:
                                    sl_distance = -100
                                elif close_price >= position_data["tp"]:
                                    tp_distance = 100
                                else:
                                    tp_distance = ((close_price - position_data["open_price"]) / 
                                                 (position_data["tp"] - position_data["open_price"])) * 100
                            else:  # sell
                                if close_price >= position_data["sl"]:
                                    sl_distance = -100
                                elif close_price <= position_data["tp"]:
                                    tp_distance = 100
                                else:
                                    tp_distance = ((position_data["open_price"] - close_price) / 
                                                 (position_data["open_price"] - position_data["tp"])) * 100
                            
                            # Report trade result
                            if position_data.get("trade_id"):
                                try:
                                    requests.post(
                                        f"{self.api_endpoint}/trade-result",
                                        json={
                                            "trade_id": position_data["trade_id"],
                                            "profit": profit,
                                            "sl_distance": sl_distance,
                                            "tp_distance": tp_distance
                                        },
                                        timeout=5
                                    )
                                except Exception as e:
                                    logger.error(f"Error reporting trade result: {str(e)}")
                    
                    del self.active_positions[ticket]
            
            except Exception as e:
                logger.error(f"Error monitoring trades: {str(e)}")
            
            time.sleep(5)
    
    def run(self):
        """Run the trading client"""
        self.running = True
        
        # Start trade monitoring thread
        import threading
        monitor_thread = threading.Thread(target=self.monitor_trades)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        logger.info("Trading client started")
        
        try:
            while self.running:
                current_time = datetime.now()
                
                if (self.last_analysis_time is None or 
                    (current_time - self.last_analysis_time).total_seconds() >= self.check_interval):
                    
                    signal = self.analyze_market()
                    if signal and signal["action"] in ["BUY", "SELL"]:
                        self.execute_trade_signal(signal)
                    
                    self.last_analysis_time = current_time
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        
        finally:
            self.running = False
            logger.info("Trading client stopped")

if __name__ == "__main__":
    client = FibonacciTradingClient('config.ini')
    if client.initialize():
        client.run()
    else:
        logger.error("Failed to initialize client")