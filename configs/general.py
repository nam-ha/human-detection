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
    port = int(os.getenv('PORT'))

@dataclass
class PathsConfig:
    configs_folder = 'configs'
    datasets_folder = 'datasets'
    outputs_folder = 'outputs'
    models_folder = 'models'

env_config = EnvConfig()
paths_config = PathsConfig()
