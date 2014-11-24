__author__ = 'christian'


from os import listdir
from os.path import isfile, join


def get_files_from_directory(dir_path, filters=[], start_with=None):
    if not filters:
        if not start_with:
            onlyfiles = [[join(dir_path, f), f]
                            for f in listdir(dir_path)
                            if isfile(join(dir_path, f))]
        else:
            onlyfiles = [[join(dir_path, f), f]
                            for f in listdir(dir_path)
                            if (isfile(join(dir_path, f)) and f.startswith(start_with))]
    else:
        if not start_with:
            onlyfiles = [[join(dir_path, f), f]
                    for f in listdir(dir_path)
                    if (isfile(join(dir_path, f)) and f.split('.')[1] in filters)]
        else:
            onlyfiles = [[join(dir_path, f), f]
                    for f in listdir(dir_path)
                    if (isfile(join(dir_path, f)) and f.split('.')[1] in filters and f.startswith(start_with))]

    return onlyfiles


