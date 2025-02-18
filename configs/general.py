import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(__file__))
)

from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(
    dotenv_path = '.env'
)

@dataclass
class EnvConfig:
    # Deployment
    port = int(os.getenv('PORT'))
    
    model_filename = os.getenv('MODEL_FILENAME')
    database_url = os.getenv('DATABASE_URL')
    
@dataclass
class PathsConfig:
    configs_folder = 'configs'
    datasets_folder = 'datasets'
    outputs_folder = 'outputs'
    models_folder = 'models'
    media_storage_folder = 'media_storage'

env_config = EnvConfig()
paths_config = PathsConfig()
