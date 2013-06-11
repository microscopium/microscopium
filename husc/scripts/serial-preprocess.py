#!/usr/bin/env python

import argparse

import numpy as np
from skimage import io as skimio
from husc import io, preprocess


parser = argparse.ArgumentParser(help="basic HUSC pipeline.")
parser.add_argument('images', nargs='+', help='input images')
parser.add_argument('-r', '--radius', type=float,
                    help="radius for illumination estimation.")


def main():
    args = parser.parse_args()
    stitch_images = preprocess.run_quadrant_stitch(args.images)
    channels = preprocess.group_by_channel(stitch_images)
    for channel_id, channel_fns in channels:
        im_iter = (skimio.imread(fn) for fn in channel_fns)
        illum = preprocess.find_background_illumination(im_iter)
        print "illumination range: ", illum.min(), illum.max()
        illum_fn = '_'.join(channel_id) + '_illum.tif'
        io.imwrite(np.round(illum).astype(np.uint16), illum_fn)
        im_iter = (skimio.imread(fn) for fn in channel_fns)
        for fn in im_iter:
            im = skimio.imread(fn)
            im = im.astype(float) / illum
            io.imwrite(np.round(im).astype(np.uint16), fn[:-4] + '_illum.tif')
        print "channel", channel_id, "done!"


if __name__ == '__main__':
    main()

