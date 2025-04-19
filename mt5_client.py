import MetaTrader5 as mt5
import requests
import time
import logging
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MT5Client")

class TradingClient:
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.session = self._create_session()
        
    def _create_session(self):
        session = requests.Session()
        
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
            respect_retry_after_header=True
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
            pool_block=True
        )
        
        session.mount("https://", adapter)
        return session

    def connect_mt5(self) -> bool:
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False
            
        logger.info(f"Connected to MT5: {mt5.terminal_info()}")
        return True

    def send_analysis_request(self, market_data: dict) -> dict:
        for attempt in range(5):
            try:
                response = self.session.post(
                    f"{self.api_endpoint}/analyze",
                    json={
                        "api_key": self.api_key,
                        **market_data
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=(3.05, 30)  # Connect and read timeouts
                )
                
                if response.status_code == 200:
                    return response.json()
                    
                if response.status_code == 503:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"Service unavailable, retrying after {retry_after}s")
                    time.sleep(retry_after)
                    continue
                    
                response.raise_for_status()
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt+1}/5): {str(e)}")
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 10))  # Exponential backoff
                continue
                
        raise ConnectionError("Failed to get analysis after 5 attempts")

    def run(self):
        logger.info("Starting trading client")
        while True:
            try:
                market_data = self.get_market_data()
                if market_data:
                    result = self.send_analysis_request(market_data)
                    self.process_signal(result)
                    
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(10)

if __name__ == "__main__":
    client = TradingClient(
        api_endpoint="https://fibonacci-trading-487611615001.us-central1.run.app",
        api_key="your-api-key"
    )
    
    if client.connect_mt5():
        client.run()