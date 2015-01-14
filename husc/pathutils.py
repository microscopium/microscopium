import os
from fnmatch import fnmatch

def all_matching_files(path, glob='*.TIF', full=True, sort=True):
    """Return list of files in directory with specified extension.

    Note: This function recurses all subdirectories.

    Parameters
    ----------
    path : string
        A directory containing files.
    glob : string
        Pattern used to match files. eg. '*.TIF' will return all
        TIF files.
    full : bool, optional
        Whether or not to return files with the path included.
    sort : bool, optional
        Whether or not to sort the list of files before returning them.
    """
    fns = []
    if full is True:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in [f for f in filenames if fnmatch(f, glob)]:
                fns.append(os.path.join(dirpath, filename))
    else:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in [f for f in filenames if fnmatch(f, glob)]:
                fns.append(filename)

    if sort is True:
        fns.sort()

    return fns
