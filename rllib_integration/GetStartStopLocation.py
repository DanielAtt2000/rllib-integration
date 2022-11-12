import pandas as pd
import random

[33,28, 27, 17,  14, 11, 10, 5]
spawn_points = pd.DataFrame(data={'N_IN':[11,33], 'N_OUT':[18,32],
                               'W_IN': [5,17], 'W_OUT': [15,34],
                               'S_IN':[27,28], 'S_OUT':[2,21],
                               'E_IN':[10,14], 'E_OUT':[4,31]
                               })


def get_entry_exit_spawn_point_indices(debug=False):
    number_of_exists = int(len(spawn_points.axes[1]) / 2)
    entry_idx = random.randint(0, number_of_exists-1) * 2
    exit_number = random.randint(1,number_of_exists)
    exit_idx = int((entry_idx + (exit_number * 2) + 1) % (number_of_exists * 2))


    entry_name = spawn_points.iloc[:, entry_idx].name
    exit_name = spawn_points.iloc[:, exit_idx].name

    assert 'IN' in entry_name
    assert 'OUT' in exit_name

    entry_spawn_point_index = spawn_points.iloc[:, entry_idx].sample().array[0]
    exit_spawn_point_index = spawn_points.iloc[:, exit_idx].sample().array[0]

    if debug:
        print(f"Exit number {exit_number}")
        print(f"Starting from {entry_name}  with index {entry_spawn_point_index}")
        print(f"Stopping at {exit_name} with index {exit_spawn_point_index}")
        print('=====================')

    return entry_spawn_point_index, exit_spawn_point_index


for _ in range(1000):
    get_entry_exit_spawn_point_indices()