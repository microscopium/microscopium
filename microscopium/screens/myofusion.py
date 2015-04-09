"""Feature computations and other functions for the Marcelle myoblast
fusion screen.
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import collections as coll

import numpy as np
from scipy import ndimage as nd
from skimage.filters import threshold_otsu, threshold_adaptive

from .. import features
from six.moves import map
from six.moves import zip


def feature_vector_from_rgb(image, sample_size=None):
    """Compute a feature vector from the composite images.

    The channels are assumed to be in the following order:
     - Red: MCF-7
     - Green: cytoplasm GFP
     - Blue: DAPI/Hoechst

    Parameters
    ----------
    im : array, shape (M, N, 3)
        The input image.
    sample_size : int, optional
        For features based on quantiles, sample this many objects
        rather than computing full distribution. This can considerably
        speed up computation with little cost to feature accuracy.

    Returns
    -------
    fs : 1D array of float
        The features of the image.
    names : list of string
        The feature names.
    """
    all_fs, all_names = [], []
    ims = np.rollaxis(image[..., :3], -1, 0) # toss out alpha chan if present
    mcf, cells, nuclei = ims
    prefixes = ['mcf', 'cells', 'nuclei']
    for im, prefix in zip(ims, prefixes):
        fs, names = features.intensity_object_features(im,
                                                       sample_size=sample_size)
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
    cells_t_adapt = threshold_adaptive(cells, 51)
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
        NOTE: the 'plate' key is converted to an int.

    Examples
    --------
    >>> fn = ('MYORES-p1-j01-110210_02490688_53caa10e-ac15-4166-9b9d-'
    ...       '4b1167f3b9c6_C04_s1_w1.TIF')
    >>> d = myores_semantic_filename(fn)
    >>> d
    OrderedDict([('directory', ''), ('prefix', 'MYORES'), ('pass', 'p1'), ('job', 'j01'), ('date', '110210'), ('plate', 2490688), ('barcode', '53caa10e-ac15-4166-9b9d-4b1167f3b9c6'), ('well', 'C04'), ('quadrant', 's1'), ('channel', 'w1'), ('suffix', 'TIF')])
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
    try:
        semantic['plate'] = int(semantic['plate'])
    except ValueError: # Some plates are labeled "NOCODE"
        semantic['plate'] = np.random.randint(1000000)
    return semantic


def filename2coord(fn):
    """Obtain (plate, well) coordinates from a filename.

    Parameters
    ----------
    fn : string
        The input filename

    Returns
    -------
    coord : (int, string) tuple
        The (plate, well) coordinates of the image.

    Examples
    --------
    >>> fn = ('MYORES-p1-j01-110210_02490688_53caa10e-ac15-4166-9b9d-'
    ...       '4b1167f3b9c6_C04_s1_w1.TIF')
    >>> filename2coord(fn)
    (2490688, 'C04')
    """
    sem = myores_semantic_filename(fn)
    return (sem['plate'], sem['well'])


def dir2plate(dirname):
    """Return a Plate ID from a directory name.
    
    Parameters
    ----------
    dirname : string
        A directory containing export images from an HCS plate.

    Returns
    -------
    plateid : int
        The plate ID parsed from the directory name.
    """
    basedir = os.path.split(dirname)[1]
    plateid = basedir.split('_')[1]
    try:
        plateid = int(plateid)
    except ValueError:
        print("Plate ID %s cannot be converted to int, replaced with 0." %
              plateid)
        return 0
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
    plate2dir_dict : {int: string}
        A dictionary mapping plates to directories.

    Returns
    -------
    fn_out : string
        The full path (from the base directory) to the filename in its
        original location.
    """
    base_fn = os.path.split(fn)[-1]
    sem = myores_semantic_filename(base_fn)
    try:
        d = plate2dir_dict[sem['plate']]
    except KeyError:
        print((fn, sem))
        raise
    return os.path.join(d, base_fn)


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
            fin.readline()
        for line in fin:
            line = line.rstrip().split(delim)
            symbol, plate, well, ctrl = [line[i] for
                                         i in symbol_plate_well_ctrl]
            plate = int(plate)
            if symbol == '' and ctrl.lower() != 'sample':
                symbol = 'control-' + ctrl
            gene2wells.setdefault(symbol, []).append((plate, well))
    return gene2wells


def make_well2file_dict(data):
    """Create a dictionary mapping wells to image files.

    Parameters
    ----------
    data : pandas data frame
        A data frame from feature computation.

    Returns
    -------
    well2file : dictionary, {(int, string): string}
        A dictionary keyed by a (plate, well) tuple whose values are
        file paths.
    """
    filenames = data.index
    filepaths = map(scratch2real, filenames)
    file_data = map(myores_semantic_filename, filepaths)
    well2file = {}
    for info, path in zip(file_data, filepaths):
        well2file[(info['plate'], info['well'])] = path
    return well2file


def make_gene2files_dict(gene2wells, well2file):
    """Create a dictionary mapping genes to images.

    Parameters
    ----------
    gene2wells : dict, {string: [(int, string)]}
        A dictionary mapping genes to wells.
    well2file : dict, {(int, string): string}
        A dictionary mapping wells to files.

    Returns
    -------
    gene2files : dict, {string, [string]}
        A dictionary mapping genes to files.
    """
    gene2files = {}
    for gene, wells in gene2wells.items():
        gene2files[gene] = [well2file(well) for well in wells]
    return gene2files


def key2mongo(tup):
    """Return a string given an (int, string) plate-well key.

    Parameters
    ----------
    tup : (int, string) tuple
        A (plate, well) identifier.

    Returns
    -------
    mongo_id : string
        A string, suitable for representing a mongodb _id field.

    Examples
    --------
    >>> tup = (2490688, 'C04')
    >>> key2mongo(tup)
    '2490688-C04'
    """
    return str(tup[0]) + '-' + tup[1]


def mongo2key(mongo_id):
    """Return an (int, string) plate-well key, given its string.

    Parameters
    ----------
    mongo_id : string
        A string representing a plate-well key, separated by a dash.

    Returns
    -------
    tup : (int, string) tuple
        The plate-well representation.

    Examples
    --------
    >>> mongo_id = "2490688-C04"
    >>> mongo2key(mongo_id)
    (2490688, 'C04')
    """
    tup = mongo_id.split('-')
    tup[0] = int(tup[0])
    return tuple(tup)


def populate_db(gene_table_filename, image_filenames, db="myofusion",
                coll_name="wells", host='localhost', port=27017):
    """Populate a MongoDB database with gene entries from the screen.

    Parameters
    ----------
    gene_table_filename : string
        The file containing the mapping of genes to plates and wells.
    image_filenames : list of string
        The filenames of images in the screen.
    db : string, optional
        The name of the database in the MongoDB server.
    coll_name : string, optional
        The name of the collection within the database to hold the
        gene data.
    host : string, optional
        The server hosting the MongoDB daemon.
    port : int, optional
        The port on which to access the MongoDB daemon.
    """
    key2doc = {}
    for filename in image_filenames:
        sem = myores_semantic_filename(filename)
        key = (sem['plate'], sem['well'])
        key2doc[key] = {'filename': filename, '_id': key2mongo(key)}
    with open(gene_table_filename, 'r') as fin:
        column_names = fin.readline().rstrip().split(',')
        idx_plate = column_names.index('cell_plate_barcode')
        idx_well = column_names.index('well')
        for line in fin:
            line = line.rstrip().split(',')
            line[idx_plate] = int(line[idx_plate])
            key = (line[idx_plate], line[idx_well])
            doc = dict(zip(column_names, line))
            if key in key2doc:
                key2doc[key].update(doc)
            else:
                key2doc[key] = doc
                key2doc[key]['_id'] = key2mongo(key)
    from pymongo import MongoClient
    collection = MongoClient(host, port)[db][coll_name]
    for doc in key2doc.values():
        collection.save(doc)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

