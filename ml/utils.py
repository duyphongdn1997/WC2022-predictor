import pickle


def write_pickle(path, a):
    """

    Args:
        path: The path to storge *.pkl file
        a: An object

    Returns:

    """
    try:
        with open(path, 'wb') as handle:
            pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as e:
        print(e)
        return False


def load_pickle(path):
    """

    Args:
        path:

    Returns:

    """
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data
