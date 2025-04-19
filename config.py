import os
import configparser
from google.cloud import secretmanager
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self._load_config()
        self._load_secrets()
        self._validate()

    def _load_config(self):
        config_locations = [
            Path('config.ini'),
            Path('/etc/fibonacci/config.ini'),
            Path.home() / '.fibonacci' / 'config.ini'
        ]
        
        for location in config_locations:
            if location.exists():
                self.config.read(location)
                logger.info(f"Loaded config from {location}")
                return
        raise FileNotFoundError("No configuration file found")

    def _load_secrets(self):
        if 'API_KEY' in os.environ:
            self.config['API']['key'] = os.environ['API_KEY']
        elif 'GCP_PROJECT_ID' in os.environ:
            try:
                secret_client = secretmanager.SecretManagerServiceClient()
                secret_name = f"projects/{os.environ['GCP_PROJECT_ID']}/secrets/api-key/versions/latest"
                response = secret_client.access_secret_version(name=secret_name)
                self.config['API']['key'] = response.payload.data.decode('UTF-8')
            except Exception as e:
                logger.error(f"Failed to load API key: {str(e)}")
                raise

    def _validate(self):
        required = {
            'API': ['endpoint', 'key'],
            'Trading': ['symbol', 'timeframe', 'candles_lookback', 'check_interval'],
            'Retry': ['max_retries', 'base_delay']
        }
        
        for section, keys in required.items():
            if section not in self.config:
                raise ValueError(f"Missing section: {section}")
            for key in keys:
                if key not in self.config[section]:
                    raise ValueError(f"Missing key: {section}.{key}")

    @property
    def api_endpoint(self):
        return self.config['API']['endpoint']

    @property
    def api_key(self):
        return self.config['API']['key']

    @property
    def trading_symbol(self):
        return self.config['Trading']['symbol']

    @property
    def timeframe(self):
        return self.config['Trading']['timeframe']

    @property
    def candles_lookback(self):
        return int(self.config['Trading']['candles_lookback'])

    @property
    def check_interval(self):
        return int(self.config['Trading']['check_interval'])

    @property
    def max_retries(self):
        return int(self.config['Retry']['max_retries'])

    @property
    def base_delay(self):
        return int(self.config['Retry']['base_delay'])

config = Config()