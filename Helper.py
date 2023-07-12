import pickle
def save_to_pickle(filename, data):
    filename = filename + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def open_pickle(filename):
    with open(filename + '.pickle', 'rb') as handle:
        return pickle.load(handle)
