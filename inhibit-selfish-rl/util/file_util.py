import os


def next_numbered_file(file="%s", return_number=False):
    """Return the next numbered file in a directory.
    Non-existing directories will be automatically created.

    Args:
        :param file: The file name to use. %s will be replaced with the number.
        :param return_number: If True, returns the file name and the number used.

    Returns the first file that doesn't exist.
    """

    directory = os.path.dirname(file)
    file = os.path.basename(file)

    assert "%s" in file, "File name template must contain %s"

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    i = 0
    while True:
        path = os.path.join(directory, file % i)
        if not os.path.exists(path):
            if return_number:
                return path, i
            return path
        i += 1
        assert i < 1000, "Exceeded 1000 numbered files."
