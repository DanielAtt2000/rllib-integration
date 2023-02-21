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
# For 4 way roundabout
spawn_points = pd.DataFrame(data={'N_IN':[17], 'N_OUT':[10],
                               'W_IN': [16], 'W_OUT': [14],
                              'S_IN':[15], 'S_OUT':[19],
                               'E_IN':[7], 'E_OUT':[6]
                               })

spawn_points_2_lane_roundabout_small = [
    [18,[38,13]],
    [84,[12,33,81]],
    [57,[13,30]],
    [39,[33,81,37]],
    [51, [81,37,12]],
    [11, [30,69]],
    [42, [37,12,33]],
    [45, [69,38]],
]

spawn_points_2_lane_roundabout_large = [
    [12,[8,51]],
    [13,[40,73]],
    [26,[51,21]],
    [82,[73,11]],
    [25,[21,8]],
    [52,[11,40]],
]

roundabouts = [spawn_points_2_lane_roundabout_small,spawn_points_2_lane_roundabout_large]


def get_entry_exit_spawn_point_indices_2_lane(failed_spawn_locations, run_through_all=False, roundabout_idx=-1, route_idx=-1, exit_idx=-1):
    entry_spawn_point_index = -1
    exit_spawn_point_index = -1


    while entry_spawn_point_index in failed_spawn_locations:
        roundabout_choice = random.choice(roundabouts)
        element = random.choice(roundabout_choice)
        entry_spawn_point_index = element[0]
        exit_spawn_point_index = random.choice(element[1])

        # Only to test in straight line
        # entry_spawn_point_index = 34
        # exit_spawn_point_index = -1
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
    if run_through_all:
        roundabout = roundabouts[roundabout_idx]
        route = roundabout[route_idx]
        entry = route[0]
        exit = route[1][exit_idx]

        exit_idx += 1

        if len(route[1]) == exit_idx:
            route_idx +=1
        if len(roundabout) == route_idx:
            roundabout_idx +=1
        if len(roundabouts) == roundabout_idx:
            raise Exception('FINISHED ALL POINTS')
        return entry, exit, roundabout_idx, route_idx, exit_idx
    else:
        return entry_spawn_point_index, exit_spawn_point_index



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
        # entry_spawn_point_index = 34
        # exit_spawn_point_index = -1
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