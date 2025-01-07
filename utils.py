import os


def check_file(path):
    if os.path.isfile(path):
        return True
    return False
