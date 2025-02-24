import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)

import json
import requests

from configs.general import env_config, paths_config

def main():
    # Load config
    config_file = os.path.join(
        paths_config.configs_folder,
        'script', 'api', 'invoke_history_config.json'
    )
    
    with open(config_file, 'r') as file:
        config = json.load(file)
        
    # Predict
    response = requests.get(
        url = f'http://localhost:{env_config.port}/api/v1/history',
        json = {
            'page_size': config.get('page_size', 10),
            'page_index': config.get('page_index', 1),
            'search_query_id': config.get('search_query_id', None),
            
            'time_min': config.get('time_min', None),
            'time_max': config.get('time_max', None),
            'num_humans_min': config.get('num_humans_min', None),
            'num_humans_max': config.get('num_humans_max', None)
        }
    )
    
    records = response.json().get('records')
    total = response.json().get('total')
    
    print("Records: ", records)
    print("Total: ", total)


if __name__ == '__main__':
    main()
