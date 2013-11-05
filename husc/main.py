#!/bin/env python

# standard library
import os
import sys
import argparse
import itertools as it

# dependencies
import mahotas as mh
from skimage import io

# local imports
from . import preprocess as pre


parser = argparse.ArgumentParser(description="Run the HUSC functions.")
subpar = parser.add_subparsers()


crop = subpar.add_parser('crop', help="Crop images.")
crop.add_argument('crops', nargs=4, metavar='INT',
                  help='xstart, xstop, ystart, ystop. "None" also allowed.')
crop.add_argument('images', nargs='+', metavar='IM', help="The input images.")
crop.add_argument('-o', '--output-suffix',
                  default='.crop.tif', metavar='SUFFIX',
                  help="What suffix to attach to the cropped images.")


mask = subpar.add_parser('mask', help="Estimate a mask over image artifacts.")
mask.add_argument('images', nargs='+', metavar='IM', help="The input images.")
mask.add_argument('-o', '--offset', metavar='INT', default=0, type=int,
                  help='Offset the automatic mask threshold by this amount.')
mask.add_argument('-v', '--verbose', action='store_true',
                  help='Print runtime information to stdout.')


illum = subpar.add_parser('illum',
                          help="Estimate and correct illumination.")
illum.add_argument('images', nargs='*', metavar='IMG', default=[],
                   help="The input images.")
illum.add_argument('-f', '--file-list', type=lambda x: open(x, 'r'),
                   metavar='FN',
                   help='Text file with one image filename per line.')
illum.add_argument('-o', '--output-suffix',
                   default='.illum.tif', metavar='SUFFIX',
                   help="What suffix to attach to the corrected images.")
illum.add_argument('-l', '--stretchlim', metavar='[0.0-1.0]', type=float,
                   default=0.0, help='Stretch image range before all else.')
illum.add_argument('-L', '--stretchlim-output', metavar='[0.0-1.0]', type=float,
                   default=0.0, help='Stretch image range before output.')
illum.add_argument('-q', '--quantile', metavar='[0.0-1.0]', type=float, 
                   default=0.05,
                   help='Use this quantile to determine illumination.')
illum.add_argument('-r', '--radius', metavar='INT', type=int, default=51,
                   help='Radius in which to find quantile.')
illum.add_argument('-s', '--save-illumination', metavar='FN',
                   help='Save the illumination field to a file.')
illum.add_argument('-v', '--verbose', action='store_true',
                   help='Print runtime information to stdout.')


stitch = subpar.add_parser('stitch', 
                            help="Stitch images by quadrant.")
stitch.add_argument('images', nargs='*', metavar='IM',
                    help="The input images.")


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
    if cmd == 'crop':
        run_crop(args)
    elif cmd == 'mask':
        run_mask(args)
    elif cmd == 'illum':
        run_illum(args)
    elif cmd == 'stitch':
        run_stitch(args)


def run_crop(args):
    """Run image cropping."""
    crops = []
    for c in args.crops:
        try:
            crops.append(int(c))
        except ValueError:
            crops.append(None)
    xstart, xstop, ystart, ystop = crops
    slices = (slice(xstart, xstop), slice(ystart, ystop))
    for imfn in args.images:
        im = mh.imread(imfn)
        imout = pre.crop(im, slices)
        fnout = os.path.splitext(imfn)[0] + args.output_suffix
        mh.imsave(fnout, imout)


def run_mask(args):
    """Run mask generation."""
    n, m = pre.write_max_masks(args.images, offset=args.offset)
    if args.verbose:
        print("%i masks created out of %i images processed" % (n, m))


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
    if args.file_list is not None:
        args.images.extend([fn.rstrip() for fn in args.file_list])
    il = pre.find_background_illumination(args.images, args.radius,
                                          args.quantile, args.stretchlim)
    if args.verbose:
        print 'illumination field:', type(il), il.dtype, il.min(), il.max()
    if args.save_illumination is not None:
        io.imsave(args.save_illumination, il / il.max())
    base_fns = (os.path.splitext(fn)[0] for fn in args.images)
    ims_out = (fn + args.output_suffix for fn in base_fns)
    ims = (mh.imread(fn) for fn in args.images)
    for im, fout in it.izip(ims, ims_out):
        im = pre.correct_image_illumination(im, il)
        if args.stretchlim_output > 0:
            lim = args.stretchlim_output
            im = pre.stretchlim(im, lim, 1 - lim)
        io.imsave(fout, im)


def run_stitch(args):
    """Run stitching.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments parsed by the argparse library.

    Returns
    -------
    None
    """
    pre.run_quadrant_stitch(args.images)


if __name__ == '__main__':
    main()

