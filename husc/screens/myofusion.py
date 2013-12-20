"""Feature computations and other functions for the Marcelle myoblast
fusion screen.
"""

import os
import collections as coll

import numpy as np
from scipy import ndimage as nd
from skimage.filter import threshold_otsu, threshold_adaptive

from .. import features


def feature_vector_from_rgb(image):
    """Compute a feature vector from the composite images.

    The channels are assumed to be in the following order:
     - Red: MCF-7
     - Green: cytoplasm GFP
     - Blue: DAPI/Hoechst

    Parameters
    ----------
    im : array, shape (M, N, 3)
        The input image.

    Returns
    -------
    fs : 1D array of float
        The features of the image.
    names : list of string
        The feature names.
    """
    all_fs, all_names = [], []
    ims = np.rollaxis(image, -1, 0)
    mcf, cells, nuclei = ims
    prefixes = ['mcf', 'cells', 'nuclei']
    for im, prefix in zip(ims, prefixes):
        fs, names = features.intensity_object_features(im)
        names = [prefix + '-' + name for name in names]
        all_fs.append(fs)
        all_names.extend(names)
    nuclei_mean = nd.label(nuclei > np.mean(nuclei))[0]
    fs, names = features.nearest_neighbors(nuclei_mean)
    all_fs.append(fs)
    all_names.extend(['nuclei-' + name for name in names])
    mcf_mean = nd.label(mcf)[0]
    fs, names = features.fraction_positive(nuclei_mean, mcf_mean,
                                           positive_name='mcf')
    all_fs.append(fs)
    all_names.extend(names)
    cells_t_otsu = cells > threshold_otsu(cells)
    cells_t_adapt = cells > threshold_adaptive(cells, 51)
    fs, names = features.nuclei_per_cell_histogram(nuclei_mean, cells_t_otsu)
    all_fs.append(fs)
    all_names.extend(['otsu-' + name for name in names])
    fs, names = features.nuclei_per_cell_histogram(nuclei_mean, cells_t_adapt)
    all_fs.append(fs)
    all_names.extend(['adapt-' + name for name in names])
    return np.concatenate(all_fs), all_names


feature_map = feature_vector_from_rgb


def myores_semantic_filename(fn):
    """Split a MYORES filename into its annotated components.

    Parameters
    ----------
    fn : string
        A filename from the MYORES high-content screening system.

    Returns
    -------
    semantic : collections.OrderedDict {string: string}
        A dictionary mapping the different components of the filename.

    Examples
    --------
    >>> fn = ('MYORES-p1-j01-110210_02490688_53caa10e-ac15-4166-9b9d-'
              '4b1167f3b9c6_C04_s1_w1.TIF')
    >>> d = myores_semantic_filename(fn)
    >>> d
    OrderedDict([('directory', ''), ('prefix', 'MYORES'), ('pass', 'p1'), ('job', 'j01'), ('date', '110210'), ('plate', '02490688'), ('barcode', '53caa10e-ac15-4166-9b9d-4b1167f3b9c6'), ('well', 'C04'), ('quadrant', 's1'), ('channel', 'w1'), ('suffix', '')])
    """
    keys = ['directory', 'prefix', 'pass', 'job', 'date', 'plate',
            'barcode', 'well', 'quadrant', 'channel', 'suffix']
    directory, fn = os.path.split(fn)
    filename, suffix = fn.split('.')[0], '.'.join(fn.split('.')[1:])
    values = filename.split('_')
    full_prefix = values[0].split('-')
    if len(full_prefix) > 4:
        head, tail = full_prefix[:3], full_prefix[3:]
        full_prefix = head + ['-'.join(tail)]
    values = [directory] + full_prefix + values[1:] + [suffix]
    semantic = coll.OrderedDict(zip(keys, values))
    return semantic


def dir2plate(dirname):
    """Return a Plate ID from a directory name.
    
    Parameters
    ----------
    dirname : string
        A directory containing export images from an HCS plate.

    Returns
    -------
    plateid : string
        The plate ID parsed from the directory name.
    """
    basedir = os.path.split(dirname)[1]
    plateid = basedir.split('_')[1]
    return plateid


def make_plate2dir_dict(dirs, base_dir='marcelle/raw-data'):
    """Return map from plate IDs to directories containing plate images.

    Parameters
    ----------
    plateids : list of string
        The plate IDs to be mapped.
    base_dir : string, optional
        The base location on the directories on the server.

    Returns
    -------
    plate2dir : {string: string}
        Dictionary mapping plate IDs to directory paths.
    """
    plate2dir = dict([(dir2plate(d), os.path.join(base_dir, d)) for d in dirs])
    return plate2dir


def scratch2real(fn, plate2dir_dict):
    """Get the full path from the image filename.

    Parameters
    ----------
    fn : string
        The input filename, stemming from the image file used in the
        cluster job (i.e. in scratch storage).
    plate2dir_dict : string
        A dictionary mapping plates to directories.

    Returns
    -------
    fn_out : string
        The full path (from the base directory) to the filename in its
        original location.
    """
    fn1 = os.path.split(fn)[-1]
    sem = myores_semantic_filename(fn1)
    try:
        d = plate2dir_dict[sem['plate']]
    except KeyError:
        print((fn, sem))
        raise
    return os.path.join(d, fn1)


# annotation file columns:
#     [(0, 'gene_name'),
#      (1, 'gene_acc'),
#      (2, 'source_plate_barcode'),
#      (3, 'source_plate_label'),
#      (4, 'cell_plate_barcode'),
#      (5, 'cell_plate_label'),
#      (6, 'well'),
#      (7, 'row'),
#      (8, 'column'),
#      (9, 'label'),
#      (10, 'experimental_content_type_name'),
#      (11, 'molecule_design_id')]


def make_gene2wells_dict(fn, delim=',', header=True,
                        symbol_plate_well_ctrl=[0, 4, 6, 10]):
    """Produce a gene to well map from the annotation table file.

    Parameters
    ----------
    fn : string
        The input filename. The file should be a comma or tab delimited
        table.
    delim : string, optional
        The delimiter between columns in a row entry.
    header : bool, optional
        Whether the file contains a header of column names.
    symbol_plate_well_ctrl : list of int, optional
        Which columns are required for this application: gene symbol,
        plate number, well, control or sample.

    Returns
    -------
    gene2wells : dict, {string: [(int, string)]}
        Dictionary mapping gene symbols to plate/well combinations.
        Controls are mapped as 'control-NEG', 'control-POS', and so on.
    """
    gene2wells = {}
    with open(fn, 'r') as fin:
        if header:
            _header = fin.readline()
        for line in fin:
            line = line.rstrip().split(delim)
            symbol, plate, well, ctrl = [line[i] for
                                         i in symbol_plate_well_ctrl]
            plate = int(plate)
            if symbol == '' and ctrl.lower() != 'sample':
                symbol = 'control-' + ctrl
            gene2wells.setdefault(symbol, []).append((plate, well))
    return gene2wells


