import pandas as pd
import random

# [33,28, 27, 17,  14, 11, 10, 5]
# For Laptop
#spawn_points = pd.DataFrame(data={'N_IN':[11], 'N_OUT':[32],
#                                'W_IN': [17], 'W_OUT': [15],
#                                'S_IN':[27], 'S_OUT':[2],
#                                'E_IN':[10], 'E_OUT':[4]
#                                })

# For PC
#N =[34,40,41,42,43,44]
spawn_points = pd.DataFrame(data={'N_IN':[34], 'N_OUT':[25],
                               'W_IN': [20], 'W_OUT': [12],
                              'S_IN':[15], 'S_OUT':[10],
                               'E_IN':[29], 'E_OUT':[2]
                               })


def get_entry_exit_spawn_point_indices(failed_spawn_locations,debug=False):
    entry_spawn_point_index = -1
    exit_spawn_point_index = -1

    while entry_spawn_point_index in failed_spawn_locations:
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

        # Only to test in straight line
        # entry_spawn_point_index =29
        # exit_spawn_point_index = 34
    if entry_spawn_point_index == -1 or exit_spawn_point_index == -1:
        raise Exception('Failed to find spawnable location')
    #TEMP
    # temp_entry_points = [34,40,41,42,43,44]
    # entry_spawn_point_index = random.choice(temp_entry_points)
    # exit_spawn_point_index = 10
    print(spawn_points)
    print(f"Starting from {entry_spawn_point_index} to {exit_spawn_point_index}")
    print("------N-----")
    print("W-----|----E")
    print("------S-----")
    return entry_spawn_point_index, exit_spawn_point_index


# for _ in range(1000):
#     print(get_entry_exit_map_locations())