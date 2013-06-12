#!/usr/bin/env python

import argparse

import numpy as np
import mahotas as mh
from skimage import img_as_ubyte
from husc import preprocess, io


parser = argparse.ArgumentParser(description="basic HUSC pipeline.")
parser.add_argument('images', nargs='+', help='input images')
parser.add_argument('-r', '--radius', type=int, default=51,
                    help="radius for illumination estimation.")
parser.add_argument('-o', '--output-8bit', action='store_true', default=False,
                    help="save 8 bit stitch images.")
parser.add_argument('-s', '--no-stitch', action='store_false', dest='stitch',
                    default=True, help="Assume stitching already done.")


def main():
    args = parser.parse_args()
    if args.stitch:
        stitch_images = preprocess.run_quadrant_stitch(args.images)
    else:
        stitch_images = args.images
    if args.output_8bit:
        for fn in stitch_images:
            im = mh.imread(fn).astype(float)
            im = np.round(preprocess.stretchlim(im) * 255).astype(np.uint8)
            im = im[100:-100, 250:-300]
            fout = fn[:-4] + '_8bit.tif'
            io.imsave(fout, im)
    channels = preprocess.group_by_channel(stitch_images)
    for channel_id, channel_fns in channels.items():
        im_iter = (preprocess.stretchlim(mh.imread(fn))[100:-100, 250:-300]
                   for fn in channel_fns)
        illum = preprocess.find_background_illumination(im_iter, args.radius)
        print "illumination range:", illum.min(), illum.max()
        illum_fn = '_'.join(channel_id) + '_illum.tif'
        illum_to_save = img_as_ubyte(preprocess.stretchlim(illum))
        io.imsave(illum_fn, illum_to_save)
        for fn in channel_fns:
            im = mh.imread(fn)[100:-100, 250:-300]
            im = im.astype(float) / illum
            im = preprocess.stretchlim(im)
            im = img_as_ubyte(im)
            fout = fn[:-4] + '_illum.tif'
            io.imsave(fout, im)
        print "channel", channel_id, "done!"


if __name__ == '__main__':
    main()

