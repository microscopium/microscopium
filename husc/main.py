#!/bin/env python

# standard library
import os
import sys
import argparse

# dependencies
import mahotas as mh
from skimage import io

# local imports
from . import preprocess as pre


parser = argparse.ArgumentParser(description="Run the HUSC functions.")
subpar = parser.add_subparsers()

stitch = subpar.add_parser('stitch', 
                            help="Stitch four quadrants into one image.")
stitch.add_argument('quadrant_images', nargs=4, metavar='IM',
                     help="The images for each quadrant in order: NW, NE, " +
                           "SW, SE.")
stitch.add_argument('output_image',
                     help="The filename for the stitched image.")

illum = subpar.add_parser('illum',
                          help="Estimate and correct illumination.")
illum.add_argument('images', nargs='+',
                   help="The input images.")
illum.add_argument('-o', '--output-suffix',
                   default='.illum.tif', metavar='SUFFIX',
                   help="What suffix to attach to the corrected images.")
illum.add_argument('-q', '--quantile', metavar='[0.0-1.0]', type=float, 
                   default=0.05,
                   help='Use this quantile to determine illumination.')
illum.add_argument('-r', '--radius', metavar='INT', type=int, default=51,
                   help='Radius in which to find quantile.')
illum.add_argument('-s', '--save-illumination', metavar='FN',
                   help='Save the illumination field to a file.')
illum.add_argument('-v', '--verbose', action='store_true',
                   help='Print runtime information to stdout.')


def get_command(argv):
    """Get the command name used from the command line.

    Parameters
    ----------
    argv : [string]
        The argument vector.

    Returns
    -------
    cmd : string
        The command name.
    """
    return argv[1]


def main():
    """Run the command-line interface."""
    args = parser.parse_args()
    cmd = get_command(sys.argv)
    if cmd == 'illum':
        run_illum(args)
    elif cmd == 'stitch':
        raise NotImplementedError('stitch not yet implemented.')


def run_illum(args):
    """Run illumination correction.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments parsed by the argparse library.

    Returns
    -------
    None
    """
    ims = (mh.imread(fn) for fn in args.images)
    il = pre.find_background_illumination(ims, args.radius, args.quantile)
    if args.verbose:
        print 'illumination field:', type(il), il.dtype, il.min(), il.max()
    if args.save_illumination is not None:
        io.imsave(args.save_illumination, il)
    base_fns = (os.path.splitext(fn)[0] for fn in args.images)
    ims_out = (fn + args.output_suffix for fn in base_fns)
    ims = (mh.imread(fn) for fn in args.images)
    for im, fout in zip(ims, ims_out):
        im = pre.correct_image_illumination(im, il)
        io.imsave(fout, im)


if __name__ == '__main__':
    main()

