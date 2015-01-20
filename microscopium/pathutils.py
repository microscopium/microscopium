from __future__ import absolute_import
import os
from fnmatch import fnmatch

def all_matching_files(path, glob='*.tif', case_sensitive=True, full=True, sort=True):
    """Recurse all subdirectories of path and return all files matching glob.

    Parameters
    ----------
    path : string
        A directory containing files.
    glob : string
        Pattern used to match files. eg. '*.tif' will return all
        TIF files.
    case_sensitive : bool, optional
        Case sensitivity of glob pattern.
    full : bool, optional
        Whether or not to return files with the path included.
    sort : bool, optional
        Whether or not to sort the list of files before returning them.

    Returns
    -------
    fns : list of string
        The list of matched files.
    """
    fns = []

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if case_sensitive:
                match = fnmatch(filename, glob)
            else:
                match = (fnmatch(filename, glob.lower()) or
                         fnmatch(filename, glob.upper()))
            if full:
                filename = os.path.join(dirpath, filename)
            if match:
                fns.append(filename)
    if sort:
        fns.sort()

    return fns
