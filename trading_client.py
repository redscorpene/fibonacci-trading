import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import config
import time
import logging
import pickle
from pathlib import Path
from prometheus_client import Counter, Gauge

logger = logging.getLogger(__name__)

# Metrics
API_ERRORS = Counter(
    'api_errors_total',
    'Total API errors',
    ['status_code']
)

API_LATENCY = Gauge(
    'api_request_latency_seconds',
    'API request latency'
)

class TradingClient:
    def __init__(self):
        self.session = self._create_session()
        self.cache_file = Path('market_data.cache')
        self.headers = {
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json'
        }

    def _create_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=config.max_retries,
            backoff_factor=config.base_delay,
            status_forcelist=[502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    def fetch_market_data(self):
        try:
            # Try live API first
            data = self._fetch_from_api()
            self._save_cache(data)
            return data
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            try:
                return self._load_cache()
            except Exception as cache_error:
                logger.error(f"Cache load failed: {str(cache_error)}")
                raise RuntimeError("API unavailable and no cached data available")

    def _fetch_from_api(self):
        start_time = time.time()
        try:
            response = self.session.get(
                f"{config.api_endpoint}/market-data",
                headers=self.headers,
                params={
                    'symbol': config.trading_symbol,
                    'timeframe': config.timeframe,
                    'limit': config.candles_lookback
                },
                timeout=10
            )
            response.raise_for_status()
            
            API_LATENCY.set(time.time() - start_time)
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            API_ERRORS.labels(status_code=e.response.status_code).inc()
            if e.response.status_code == 503:
                self._handle_service_unavailable()
            raise

    def _handle_service_unavailable(self):
        """Exponential backoff when API is unavailable"""
        for attempt in range(config.max_retries):
            wait_time = min(config.base_delay * (2 ** attempt), 300)
            logger.warning(f"API Unavailable. Retry {attempt+1}/{config.max_retries} in {wait_time}s")
            time.sleep(wait_time)
            
            try:
                if self._check_api_health():
                    return
            except Exception:
                continue
                
        logger.error("API unavailable after multiple retries")
        raise RuntimeError("API unavailable after multiple retries")

    def _check_api_health(self):
        try:
            response = self.session.get(
                f"{config.api_endpoint}/health",
                headers=self.headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def _save_cache(self, data):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {str(e)}")

    def _load_cache(self):
        if not self.cache_file.exists():
            raise FileNotFoundError("No cache file found")
            
        with open(self.cache_file, 'rb') as f:
            cache = pickle.load(f)
            if time.time() - cache['timestamp'] > 3600:
                logger.warning("Using stale cached data (older than 1 hour)")
            return cache['data']