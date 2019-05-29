import os
import yaml

CURDIR = os.path.dirname(os.path.abspath(__file__))


def default_yaml_path():
    return os.path.join(CURDIR, 'data', 'default-settings.yaml')


def default_config():
    with open(default_yaml_path(), 'r') as fin:
        return yaml.safe_load(fin)