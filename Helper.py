import pickle
def save_to_pickle(filename, data):
    filename = filename + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def open_pickle(filename):
    with open(filename + '.pickle', 'rb') as handle:
        return pickle.load(handle)

def append_to_txt(filename, data):
    filename = filename + '.txt'
    with open(filename,'a+') as file:
        file.write(data)

def clear_txt_file(filename):
    filename = filename + '.txt'
    with open(filename,'w') as file:
        file.write('')

def read_txt_file(filename):
    filename = filename + '.txt'
    output = None
    with open(filename,'r+') as file:
        output = file.readlines()

    return output