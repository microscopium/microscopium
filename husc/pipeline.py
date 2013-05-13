#!/bin/env python

"""
A Ruffus-based pipeline to run HUSC.
"""

# built-ins
import os

# libraries
from ruffus import transform, merge
from rubra.utils import runStageCheck as run_stage_check


