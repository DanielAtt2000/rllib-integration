import pickle
def save_to_pickle(filename, data):
    filename = filename + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

save_to_pickle('waiting_times', [0, 20, 40, 60, 80, 100, 120, 140, 160, 180])