import pandas as pd
import random

from rllib_integration.RouteGeneration.global_route_planner import GlobalRoutePlanner
from rllib_integration.TestingWayPointUpdater import plot_all_routes

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
    # [70, [75,28,35]],
    # [38, [16,7]],
    # [64, [28,35,15]],
    # [17, [7,29]],
    # [66, [35,15,75]],
    # [6, [29,52]],
    # [58, [15,75,28]],
    # [77, [52,16]]
    [17,[28,98,19]],
    # [44,[75,27]],
    [95,[98,19,26]],
    # [25,[27,107]],
    [2,[19,26,28]],
    # [108,[107,18]],
    [102,[26,28,98]],
    # [77,[75]],
#     REmoved 77 to 18 --too tight
]

spawn_points_2_lane_roundabout_large = [
    # [32, [34,4,71]],
    # [86,[10,33]],
    # [40,[4,71,11]],
    # [89,[33,68]],
    # [65,[71,11,34]],
    # [27,[68,25]],
    # [23,[11,34,4]],
    # [24,[25,10]],
    [118,[84,121,58]],
    # [97,[22,82]],
    [76,[121,58,74]],
    # [112,[9]],
    [39,[58,74,84]],
    # [41,[9,56]],
    [111,[74,84,121]],
    # [55,[56,22]],
#     Removed 112 to 82 too tight
]

# roundabouts = [spawn_points_2_lane_roundabout_large,spawn_points_2_lane_roundabout_small]
roundabouts = [spawn_points_2_lane_roundabout_small,spawn_points_2_lane_roundabout_large]


def get_entry_exit_spawn_point_indices_2_lane(failed_spawn_locations):
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

    return entry_spawn_point_index, exit_spawn_point_index

# ONLY TO BE USED FOR TESTING
# def get_entry_exit_spawn_point_indices_2_lane(failed_spawn_locations):
#     entry_spawn_point_index = -1
#     exit_spawn_point_index = -1
#
#     previous_routes_files = open('testing_routes.txt','r')
#
#     lines = previous_routes_files.readlines()
#     indices = {}
#     for line in lines:
#         if "roundabout_idx" in line:
#             indices['roundabout_idx'] = int(line.split(':')[1])
#         if "entry_idx" in line:
#             indices['entry_idx'] = int(line.split(':')[1])
#         if "exit_idx" in line:
#             indices['exit_idx'] = int(line.split(':')[1])
#
#     previous_routes_files.close()
#
#     entry_spawn_point_index = roundabouts[indices['roundabout_idx']][indices['entry_idx']][0]
#     exit_spawn_point_index = roundabouts[indices['roundabout_idx']][indices['entry_idx']][1][indices['exit_idx']]
#
#     indices['exit_idx'] += 1
#     print(f"indices['exit_idx']:{indices['exit_idx']}")
#     print(f"len(roundabouts[indices['roundabout_idx']][indices['entry_idx']][1]):{len(roundabouts[indices['roundabout_idx']][indices['entry_idx']][1])}")
#
#
#     print(f"indices['entry_idx']:{indices['entry_idx']}")
#     print(f"len(roundabouts[indices['roundabout_idx']][indices['entry_idx']]):{len(roundabouts[indices['roundabout_idx']])}")
#     if indices['exit_idx'] == len(roundabouts[indices['roundabout_idx']][indices['entry_idx']][1]):
#         indices['entry_idx'] += 1
#         indices['exit_idx'] = 0
#     if indices['entry_idx'] + 1 == len(roundabouts[indices['roundabout_idx']]):
#         indices['roundabout_idx'] += 1
#         indices['entry_idx'] = 0
#         indices['exit_idx'] = 0
#
#
#     previous_routes_files = open('testing_routes.txt', 'w')
#     previous_routes_files.write(f"roundabout_idx:{indices['roundabout_idx']}\n")
#     previous_routes_files.write(f"entry_idx:{indices['entry_idx']}\n")
#     previous_routes_files.write(f"exit_idx:{indices['exit_idx']}\n")
#     previous_routes_files.close()
#
#
#     if entry_spawn_point_index == -1 or exit_spawn_point_index == -1:
#         raise Exception('Failed to find spawnable location')
#
#     print(spawn_points)
#     print(f"Starting from {entry_spawn_point_index} to {exit_spawn_point_index}")
#     print("------N-----")
#     print("W-----|----E")
#     print("------S-----")
#
#     return entry_spawn_point_index, exit_spawn_point_index

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


def visualise_all_routes(map):
    all_routes = []

    for roundabout in roundabouts:
        for entry in roundabout:
            for exit in entry[1]:
                print(f"from {entry[0]} to {exit}")
                entry_spawn_point = map.get_spawn_points()[entry[0]]
                exit_spawn_point = map.get_spawn_points()[exit]



                # # Specify more than one starting point so the RL doesn't always start from the same position
                # spawn_point_no = random.choice([33, 28, 27, 17, 14, 11, 10, 5])
                # spawn_points = [self.map.get_spawn_points()[spawn_point_no]]

                # Obtaining the route information

                start_waypoint = map.get_waypoint(entry_spawn_point.location)
                end_waypoint = map.get_waypoint(exit_spawn_point.location)

                start_location = start_waypoint.transform.location
                end_location = end_waypoint.transform.location

                sampling_resolution = 2
                global_planner = GlobalRoutePlanner(map, sampling_resolution)

                try:
                    route_waypoints = global_planner.trace_route(start_location, end_location)

                except:
                    print(f"-----from {entry[0]} to {exit}")
                    continue
                route = []
                last_x = -1
                last_y = -1

                for route_waypoint in route_waypoints:

                    # Some waypoint may be duplicated
                    # Checking and ignoring duplicated points
                    if last_x == round(route_waypoint[0].transform.location.x, 5) and last_y == round(
                            route_waypoint[0].transform.location.y, 5):
                        continue

                    last_x = round(route_waypoint[0].transform.location.x, 5)
                    last_y = round(route_waypoint[0].transform.location.y, 5)

                    # self.route.append(carla.Transform(
                    #     carla.Location(self.normalise_map_location(route_waypoint[0].transform.location.x, 'x'),
                    #                    self.normalise_map_location(route_waypoint[0].transform.location.y, 'y'),
                    #                    0),
                    #     carla.Rotation(0, 0, 0)))

                    route.append(route_waypoint[0].transform)

                all_routes.append(route)

    plot_all_routes(all_routes=all_routes)



# for _ in range(1000):
#     print(get_entry_exit_map_locations())