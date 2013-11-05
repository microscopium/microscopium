#!/bin/env python

"""
A Ruffus-based pipeline to run HUSC.
"""

# built-ins
import os
import argparse
import itertools as it

# libraries
import numpy as np
from ruffus import transform, merge, regex, collate
# from rubra.utils import runStageCheck as run_stage_check

from skimage import io as skimio


# local files
from . import io, preprocess


parser = argparse.ArgumentParser(description='Run HUSC pipeline.')
parser.add_argument('-i', '--input-files', required=True,
                    help="Input images to process")
args = parser.parse_args()


@collate(args.input_files, regex(r'(.*)_(s[1-4])_(w[1-3]).TIF'),
         r'\1_\2.illum.tif')
def estimate_illumination(input_files, output_file, radius=51, quantile=0.05):
    illum = preprocess.find_background_illumination(input_files, radius,
                                                    quantile)
    illum = np.floor(illum).astype(np.uint16)
    io.imwrite(illum, output_file)


@merge()
def correct_illumination():
    pass
