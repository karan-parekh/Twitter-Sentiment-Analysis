import os


def app_path():
    """Returns app root"""
    return os.getcwd().rsplit(os.sep, 1)[0]


def data_path(filename: str='') -> str:
    return os.path.join(app_path(), 'data', filename)


def raw_data_path(filename=''):
    return os.path.join(app_path(), 'data', 'raw', filename)


def processed_data_path(filename=''):
    return os.path.join(app_path(), 'data', 'processed', filename)


def inter_data_path(filename=''):
    return os.path.join(app_path(), 'data', 'inter', filename)

def storage_path(filename=''):
    return os.path.join(app_path(), 'storage', filename)
