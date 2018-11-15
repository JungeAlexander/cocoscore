import gzip


def get_file_handle(file_path, compression):
    """
    Returns a file handle to the given path.

    :param file_path: path to the file to open
    :param compression: indicates whether or not the input file is compressed
    :return: a file handle to file_path
    """
    if compression:
        return gzip.open(file_path, 'rt', encoding='utf-8', errors='strict')
    else:
        return open(file_path, 'rt', encoding='utf-8', errors='strict')
