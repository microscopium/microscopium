#!/usr/bin/env python
import os

import yaml

def load_config(yaml_filename):
    with open(yaml_filename, "r") as f:
        settings = yaml.safe_load(f)
    return settings


def get_tooltips(settings):
    tooltip_columns = settings['tooltip-columns']
    return [(column, '@' + column) for column in tooltip_columns]
