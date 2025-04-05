import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if running in cloud environment
is_cloud_environment = os.environ.get('K_SERVICE') is not None

# Try to load Google Cloud credentials
try:
    import google.auth
    from google.cloud import secretmanager, firestore, storage
    google_available = True
    logger.info("Google Cloud libraries imported successfully")
except ImportError:
    google_available = False
    logger.warning("Google Cloud libraries not available")

# Initialize services
db_firestore = None
storage_client = None
secret_client = None

if google_available:
    try:
        # Try to authenticate
        if is_cloud_environment:
            # In Cloud Run, use the default service account
            project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
            if not project_id:
                # Try to get project ID from metadata
                import requests
                project_id = requests.get(
                    'http://metadata.google.internal/computeMetadata/v1/project/project-id',
                    headers={'Metadata-Flavor': 'Google'}
                ).text
                os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
        else:
            # For local development with key file
            key_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'key.json')
            if os.path.exists(key_file):
                with open(key_file, 'r') as f:
                    key_data = json.load(f)
                    project_id = key_data.get('project_id')
                    if project_id:
                        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
            else:
                project_id = None
                
        # Initialize services if project_id is available
        if project_id:
            db_firestore = firestore.Client(project=project_id)
            storage_client = storage.Client(project=project_id)
            secret_client = secretmanager.SecretManagerServiceClient()
            logger.info(f"Initialized Google Cloud services for project {project_id}")
        else:
            logger.warning("No project ID found, using local services only")
    except Exception as e:
        logger.exception(f"Error initializing Google Cloud services: {e}")

# API key handling
def get_api_key():
    # Try to get from Secret Manager first
    if secret_client and is_cloud_environment:
        try:
            project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
            name = f"projects/{project_id}/secrets/api-key/versions/latest"
            response = secret_client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.warning(f"Could not retrieve API key from Secret Manager: {e}")
    
    # Fallback to environment variable
    return os.environ.get('API_KEY', 'dev-key-for-testing')

# Use SQLite by default, Firestore in cloud if available
def get_db_client():
    if db_firestore and (is_cloud_environment or os.environ.get('USE_FIRESTORE') == '1'):
        return {'type': 'firestore', 'client': db_firestore}
    else:
        return {'type': 'sqlite', 'url': os.environ.get('DATABASE_URL', 'sqlite:///./data/trading.db')}