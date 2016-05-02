from __future__ import absolute_import, division, print_function
#!/bin/env python

# standard library
import os
import sys
import argparse
import ast

# dependencies
import numpy as np
from skimage import img_as_ubyte
import toolz as tz
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler

# local imports
from . import io
from . import screens
from .screens import cellomics
from . import preprocess as pre
from . import cluster
from .io import temporary_hdf5_dataset
from six.moves import map, zip


parser = argparse.ArgumentParser(description="Run the microscopium functions.")
subpar = parser.add_subparsers()


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
    elif cmd == 'montage':
        run_montage(args)
    elif cmd == 'features':
        run_features(args)
    else:
        sys.stderr.write("Error: command %s not found. Run %s -h for help." %
                         (cmd, sys.argv[0]))
        sys.exit(2) # 2 is commonly a command-line error


crop = subpar.add_parser('crop', help="Crop images.")
crop.add_argument('crops', nargs=4, metavar='INT',
                  help='xstart, xstop, ystart, ystop. "None" also allowed.')
crop.add_argument('images', nargs='+', metavar='IM', help="The input images.")
crop.add_argument('-o', '--output-suffix',
                  default='.crop.tif', metavar='SUFFIX',
                  help="What suffix to attach to the cropped images.")
crop.add_argument('-O', '--output-dir',
                  help="Directory in which to output the cropped images.")
crop.add_argument('-c', '--compress', metavar='INT', type=int, default=1,
                  help='Use TIFF compression in the range 0 (no compression) '
                       'to 9 (max compression, slowest) (default 1).')
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
        im = io.imread(imfn)
        imout = pre.crop(im, slices)
        fnout = os.path.splitext(imfn)[0] + args.output_suffix
        if args.output_dir is not None:
            fnout = os.path.join(args.output_dir, os.path.split(fnout)[1])
        io.imsave(fnout, imout, compress=args.compress)


mask = subpar.add_parser('mask', help="Estimate a mask over image artifacts.")
mask.add_argument('images', nargs='+', metavar='IM', help="The input images.")
mask.add_argument('-o', '--offset', metavar='INT', default=0, type=int,
                  help='Offset the automatic mask threshold by this amount.')
mask.add_argument('-v', '--verbose', action='store_true',
                  help='Print runtime information to stdout.')
mask.add_argument('-c', '--close', metavar='RADIUS', default=0, type=int,
                  help='Perform morphological closing of masks of this radius.')
mask.add_argument('-e', '--erode', metavar='RADIUS', default=0, type=int,
                  help='Perform morphological erosion of masks of this radius.')
def run_mask(args):
    """Run mask generation."""
    n, m = pre.write_max_masks(args.images, args.offset, args.close, args.erode)
    if args.verbose:
        print("%i masks created out of %i images processed" % (n, m))


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
illum.add_argument('-i', '--input-bitdepth', type=int, default=None,
                   help='Input bit-depth of images.')
illum.add_argument('-r', '--radius', metavar='INT', type=int, default=51,
                   help='Radius in which to find quantile.')
illum.add_argument('-s', '--save-illumination', metavar='FN',
                   help='Save the illumination field to a file.')
illum.add_argument('-c', '--compress', metavar='INT', type=int, default=1,
                   help='Use TIFF compression in the range 0 (no compression) '
                        'to 9 (max compression, slowest) (default 1).')
illum.add_argument('-v', '--verbose', action='store_true',
                   help='Print runtime information to stdout.')
illum.add_argument('--method', metavar='STR', default='median',
                   help='How to collapse filtered images to illumination '
                        'field. options: median (default), mean.')
illum.add_argument('--random-seed', type=int, default=None,
                   help='The random seed for sampling illumination image.')
def run_illum(args):
    """Run illumination correction.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments parsed by the argparse library.
    """
    if args.file_list is not None:
        args.images.extend([fn.rstrip() for fn in args.file_list])
    il = pre.find_background_illumination(args.images, args.input_bitdepth,
                                          args.radius, args.quantile,
                                          args.stretchlim, args.method)
    if args.verbose:
        print('illumination field:', type(il), il.dtype, il.min(), il.max())
    if args.save_illumination is not None:
        io.imsave(args.save_illumination, il / il.max())
    base_fns = [pre.basefn(fn) for fn in args.images]
    ims_out = [fn + args.output_suffix for fn in base_fns]
    corrected = pre.correct_multiimage_illumination(args.images, il,
                                                    args.stretchlim_output,
                                                    args.random_seed)
    for im, fout in zip(corrected, ims_out):
        io.imsave(fout, im, compress=args.compress)


montage = subpar.add_parser('montage',
                            help='Montage and channel stack images.')
montage.add_argument('images', nargs='*', metavar='IM',
                     help="The input images.")
montage.add_argument('-c', '--compress', metavar='INT', type=int, default=1,
                     help='Use TIFF compression in the range 0 '
                          '(no compression) '
                          'to 9 (max compression, slowest) (default 1).')
montage.add_argument('-o', '--montage-order', type=ast.literal_eval,
                     default=cellomics.SPIRAL_CLOCKWISE_RIGHT_25,
                     help='The shape of the final montage.')
montage.add_argument('-O', '--channel-order', type=ast.literal_eval,
                     default=[0, 1, 2],
                     help='The position of red, green, and blue channels '
                          'in the stream.')
montage.add_argument('-s', '--suffix', default='.montage.tif',
                     help='The suffix for saved images after conversion.')
montage.add_argument('-d', '--output-dir', default=None,
                     help='The output directory for the images. Defaults to '
                          'the input directory.')
def run_montage(args):
    """Run montaging and channel concatenation.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments parsed by the argparse library.
    """
    ims = map(io.imread, args.images)
    ims_out = pre.montage_stream(ims, montage_order=args.montage_order,
                                 channel_order=args.channel_order)
    def out_fn(fn):
        sem = cellomics.cellomics_semantic_filename(fn)
        out_fn = '_'.join([str(sem[k])
                           for k in sem
                           if k not in ['field', 'channel', 'suffix']
                           and sem[k] != ''])
        outdir = (args.output_dir if args.output_dir is not None
                  else sem['directory'])
        out = os.path.join(outdir, out_fn) + args.suffix
        return out
    step = np.array(args.montage_order).size * len(args.channel_order)
    out_fns = (out_fn(fn) for fn in args.images[::step])
    for im, fn in zip(ims_out, out_fns):
        try:
            io.imsave(fn, im, compress=args.compress)
        except ValueError:
            im = img_as_ubyte(pre.stretchlim(im, 0.001, 0.999))
            io.imsave(fn, im, compress=args.compress)


features = subpar.add_parser('features',
                             help="Map images to feature vectors.")
features.add_argument('images', nargs='*', metavar='IM',
                      help="The input images.")
features.add_argument('-s', '--screen', default='cellomics',
                      help="The name of the screen being run. Feature maps "
                           "appropriate for the screen should be in the "
                           "'screens' package.")
features.add_argument('-c', '--n-components', type=int, default=2,
                      help='The number of components to compute for PCA.')
features.add_argument('-b', '--pca-batch-size', type=int, default=384,
                      help='The number of samples needed for each step of the '
                           'incremental PCA.')
features.add_argument('-n', '--num-neighbours', type=int, default=25,
                      help='The number of nearest neighbours to output '
                           'per sample.')
features.add_argument('-S', '--sample-size', type=int, default=None,
                      help='For feature computations that depend on objects, '
                           'sample this many objects.')
features.add_argument('--random-seed', type=int, default=None,
                      help='Set random seed, for testing and debugging only.')
features.add_argument('-e', '--emitter', default='json',
                      help='Format to output features during computation.')
features.add_argument('-G', '--global-threshold', action='store_true',
                      help='Use sampled intensity from all images to obtain '
                           'a global threshold.')
def run_features(args):
    """Run image feature computation.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments parsed by the argparse library.
    """
    if args.global_threshold:
        images = map(io.imread, args.images)
        thresholds = pre.global_threshold(images, args.random_seed)
    else:
        thresholds = None
    images = map(io.imread, args.images)
    screen_info = screens.d[args.screen]
    index_function, fmap = screen_info['index'], screen_info['fmap']
    fmap = tz.partial(fmap, threshold=thresholds,
                            sample_size=args.sample_size,
                            random_seed=args.random_seed)
    indices = list(map(index_function, args.images))
    f0, feature_names = fmap(next(images))
    feature_vectors = tz.cons(f0, (fmap(im)[0] for im in images))
    online_scaler = StandardScaler()
    online_pca = cluster.OnlineIncrementalPCA(n_components=args.n_components,
                                              batch_size=args.pca_batch_size)
    nimages, nfeatures = len(args.images), len(f0)
    emit = io.emitter_function(args.emitter)
    with temporary_hdf5_dataset((nimages, nfeatures), 'float') as dset:
        # First pass: compute the features, compute the mean and SD,
        # compute the PCA
        for i, (idx, v) in enumerate(zip(indices, feature_vectors)):
            emit({'_id': idx, 'feature_vector': list(v)})
            dset[i] = v
            online_scaler.partial_fit(v.reshape(1, -1))
            online_pca.add_sample(v)
        # Second pass: standardise the feature vectors, compute PCA-transform
        for i, (idx, v) in enumerate(zip(indices, dset)):
            v_std = online_scaler.transform(v)
            v_pca = online_pca.transform(v)
            dset[i] = v_std
            emit({'_id': idx, 'feature_vector_std': list(v_std),
                              'pca_vector': list(v_pca)})
            online_pca.transform(v)
        # Third pass: Compute the nearest neighbors graph.
        # THIS ANNOYINGLY INSTANTIATES FULL ARRAY -- no out-of-core
        # solution that I'm aware of...
        ng = neighbors.kneighbors_graph(dset, args.num_neighbours,
                                        include_self=False, mode='distance')
        for idx, row in zip(indices, ng):
            emit({'_id': idx, 'neighbours': [indices[i] for i in row.indices]})


if __name__ == '__main__':
    main()

