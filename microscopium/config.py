#!/usr/bin/env python
import os

import yaml

settings_filename = os.path.join(os.path.dirname(__file__), "settings.yaml")
with open(settings_filename, "r") as f:
    settings = yaml.load(f)

cluster_methods = settings['cluster-methods']

tooltip_columns = settings['tooltip-columns']
tooltips_scatter = [tuple([column, '@' + column]) for column in tooltip_columns]

color_columns_categorical = settings['color-columns']['categorical']
color_columns_numeric = settings['color-columns']['numeric']
