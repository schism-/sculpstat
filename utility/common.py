__author__ = 'christian'


from os import listdir
from os.path import isfile, join


def get_files_from_directory(dir_path, filters=[]):
    if not filters:
        onlyfiles = [[join(dir_path, f), f]
                        for f in listdir(dir_path)
                        if isfile(join(dir_path, f))]
    else:
        onlyfiles = [[join(dir_path, f), f]
                for f in listdir(dir_path)
                if (isfile(join(dir_path, f)) and f.split('.')[1] in filters)]

    return onlyfiles


