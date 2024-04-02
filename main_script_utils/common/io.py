import os


def path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
