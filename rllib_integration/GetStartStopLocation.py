import math

import pandas as pd
import random

from rllib_integration.Circle import get_radii
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

# spawn_points_2_lane_roundabout_small = [
#     # [70, [75,28,35]],
#     # [38, [16,7]],
#     # [64, [28,35,15]],
#     # [17, [7,29]],
#     # [66, [35,15,75]],
#     # [6, [29,52]],
#     # [58, [15,75,28]],
#     # [77, [52,16]]
#     [17,[28],"left"],
#     [17,[98],"left"],
#     [17,[19],"left"],
#
#     [44,[27],"right"],
#     [44,[75],"right"],
#
#     [95,[98],"left"],
#     [95,[19],"left"],
#     [95,[26],"left"],
#
#     [25,[27],"right"],
#     [25,[107],"right"],
#
#     [2,[19],"left"],
#     [2,[26],"left"],
#     [2,[28],"left"],
#
#     [108,[107],"right"],
#     [108,[18],"right"],
#
#     [102,[26],"left"],
#     [102,[28],"left"],
#     [102,[98],"left"],
#
#     [77,[75],"right"],
# #     REmoved 77 to 18 --too tight
# ]
#
# spawn_points_2_lane_roundabout_large = [
#     # [32, [34,4,71]],
#     # [86,[10,33]],
#     # [40,[4,71,11]],
#     # [89,[33,68]],
#     # [65,[71,11,34]],
#     # [27,[68,25]],
#     # [23,[11,34,4]],
#     # [24,[25,10]],
#     [118,[84], "left"],
#     [118,[121], "left"],
#     [118,[58], "left"],
#
#     [97,[22], "right"],
#     [97,[82], "right"],
#
#     [76,[121], "left"],
#     [76,[58], "left"],
#     [76,[74], "left"],
#
#     [112,[82], "right"],
#     [112,[9], "right"],
#
#     [39,[58], "left"],
#     [39,[74], "left"],
#     [39,[84], "left"],
#
#     [41,[9], "right"],
#     [41,[56], "right"],
#
#     [111,[74], "left"],
#     [111,[84], "left"],
#     [111,[121], "left"],
#
#     [55,[56], "right"],
#     [55,[22], "right"],
# #     Removed 112 to 82 too tight
# ]
#
#
# spawn_points_2_lane_roundabout_easy = [
#     # From large roundabout
#     [118, [84, 121, 58]],
#     [97, [22, 82]],
#     [76, [121, 58, 74]],
#     [112, [9]],
#     [39, [58, 74, 84]],
#     [41, [9, 56]],
#     [111, [74, 84, 121]],
#     [55, [56, 22]],
#
#     # From small roundabout
#     [17, [28]],
#     [95, [98]],
#     [2, [19]],
#     [102, [26]],
#
#     [44, [27]],
#     [25, [107]],
#     [108, [18]],
#     [77, [75]],
#
#     [44, [75]],
#     [25, [27]],
#     [108, [107]],
# ]
#
#
#
# spawn_points_2_lane_roundabout_difficult = [
#     # From small roundabout
#     [17, [98, 19]],
#     [95, [19, 26]],
#     [2, [26, 28]],
#     [102, [28, 98]],
#
#     [77,[18]],
#
#     # From large roundabout
#     [112, [82]],
# ]

spawn_points_2_lane_roundabout_small_easy = [
    [31, [43],"left"],
    [61, [95],"right"],
    [61, [42],"right"],

    [117, [120], "left"],
    [40, [42], "right"],
    [40, [10], "right"],

    # Large gap in waypoints
    [34, [112], "left"],
    [11, [10], "right"],
    # Large gap in waypoints
    [11, [3], "right"],

    [5, [41], "left"],
    # [97, [3], "right"],
    [97, [95], "right"],

    # large roundabout# large roundabout# large roundabout
    [119, [37], "right"],
    [119, [103], "right"],
    [22, [105], "left"],
    [22, [26], "left"],
    [22, [76], "left"],

    [16, [103], "right"],
    [16, [111], "right"],
    [96, [26], "left"],
    [96, [76], "left"],

    [58, [111], "right"],
    # large roundabout# large roundabout# large roundabout


]

spawn_points_2_lane_roundabout_small_difficult = [
    # In Editor
    # [17, [98],"left"],
    # # Large gaps in waypoints
    # # [31, [112],"left"],
    #
    # # Large gaps in waypoints
    # # [117, [112],"left"],
    # [95, [26],"left"],
    #
    # [2, [26], "left"],
    # [2, [28], "left"],
    #
    # [102, [98], "left"],
    # [102, [28], "left"],
    # In Editor

    [31, [120],"left"],
    # Large gaps in waypoints
    [31, [112],"left"],

    # Large gaps in waypoints
    [117, [112],"left"],
    [117, [41],"left"],

    [34, [41], "left"],
    [34, [43], "left"],

    [5, [43], "left"],
    [5, [120], "left"],
    #

    # large roundabout# large roundabout# large roundabout
    [73, [74], "right"],
    [73, [37], "right"],
    [15, [94], "left"],
    [15, [105], "left"],
    [15, [26], "left"],

    [55, [94], "left"],
    [55, [105], "left"],
    [96, [94], "left"],
    [58, [74], "right"],
    [55, [76], "left"],
    # large roundabout# large roundabout# large roundabout


]

lower_medium_roundabout = [
    [28, [17], "left"],
    [28, [46], "left"],
    [28, [67], "left"],
    [35, [41], "right"],
    [35, [59], "right"],

    # [12, [46], "left"],
    [12, [67], "left"],
    # [12, [24], "left"],
    # [13, [59], "right"],
    # [13, [42], "right"],

    [18, [67], "left"],
    # [18, [24], "left"],
    [18, [17], "left"],
    # [69, [42], "right"],
    [69, [50], "right"],

    # [, [], "left"],
    # [, [], "left"],
    # [, [], "left"],
    # [, [], "right"],
    # [, [], "right"],

]

upper_medium_roundabout = [
    [26, [14], "left"],
    [26, [3], "left"],
    [26, [63], "left"],
    [27, [66], "right"],
    # [27, [2], "right"],

    [4, [3], "left"],
    [4, [63], "left"],
    [4, [14], "left"],
    [64, [2], "right"],
    # [64, [25], "right"],

    # [, [], "left"],
    # [, [], "left"],
    # [, [], "left"],
    # [, [], "right"],
    # [, [], "right"],

]

roundabout20m = [

    [8, [14], "right"],
    [8, [35], "right"],

    [32, [2], "left"],
    [32, [31], "left"],
    [32, [4], "left"],
    [24, [35], "right"],
    [24, [6], "right"],

    [20, [31], "left"],
    # [20, [4], "left"],
    [20, [7], "left"],
    [19, [6], "right"],
    [19, [26], "right"],

    [1, [4], "left"],
    [1, [7], "left"],
    [1, [2], "left"],
    [0, [26], "right"],
    [0, [14], "right"],

]

roundabouts = [lower_medium_roundabout,upper_medium_roundabout]
# roundabouts = [spawn_points_2_lane_roundabout_difficult,spawn_points_2_lane_roundabout_easy]


def get_entry_exit_spawn_point_indices_2_lane(failed_spawn_locations, last_roundabout_choice):
    entry_spawn_point_index = -1
    exit_spawn_point_index = -1

    if last_roundabout_choice == 0:
        last_roundabout_choice = 1
    else:
        last_roundabout_choice = 0

    if len(roundabouts) == 1:
        last_roundabout_choice = 0

    while entry_spawn_point_index in failed_spawn_locations:
        roundabout_choice = roundabouts[last_roundabout_choice]
        element = random.choice(roundabout_choice)
        entry_spawn_point_index = element[0]
        exit_spawn_point_index = random.choice(element[1])
        route_lane = element[2]

        # Only to test in straight line
        # entry_spawn_point_index = 34
        # exit_spawn_point_index = -1
    if entry_spawn_point_index == -1 or exit_spawn_point_index == -1:
        raise Exception('Failed to find spawnable location')
    #TEMP
    # temp_entry_points = [34,40,41,42,43,44]
    # entry_spawn_point_index = random.choice(temp_entry_points)
    # exit_spawn_point_index = 10
    # print(spawn_points)
    # print(f"Starting from {entry_spawn_point_index} to {exit_spawn_point_index}")
    # print("------N-----")
    # print("W-----|----E")
    # print("------S-----")

    return entry_spawn_point_index, exit_spawn_point_index, route_lane, last_roundabout_choice

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

def is_in_front_of_waypoint_from_vector(line_point,vector_line, in_front_point):
    # https://stackoverflow.com/questions/22668659/calculate-on-which-side-of-a-line-a-point-is
    # Vector equation of the line is
    #  r = point on line + t(parallel line)

    x_0 = line_point.location.x
    y_0 = line_point.location.y
    t = 2
    x_1 = x_0 + t*vector_line.x
    y_1 = y_0 + t*vector_line.y

    x_p = in_front_point.location.x
    y_p = in_front_point.location.y

    d_pos = (x_1-x_0)*(y_p-y_0) - (x_p-x_0)*(y_1-y_0)

    if d_pos == 0:
        # point on line
        return 0
    elif d_pos > 0:
        # Point behind line
        return 1
    elif d_pos < 0:
        # Point in front of line
        return -1
    else:
        raise Exception("INVALID POSITION")

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
                last_waypoint_transform = None

                for route_waypoint in route_waypoints:

                    # Some waypoint may be duplicated
                    # Checking and ignoring duplicated points
                    if abs(last_x - round(route_waypoint[0].transform.location.x, 2)) < 0.4 and abs(last_y - round(
                            route_waypoint[0].transform.location.y, 2)) < 0.4:
                        continue

                    last_x = round(route_waypoint[0].transform.location.x, 2)
                    last_y = round(route_waypoint[0].transform.location.y, 2)

                    # self.route.append(carla.Transform(
                    #     carla.Location(self.normalise_map_location(route_waypoint[0].transform.location.x, 'x'),
                    #                    self.normalise_map_location(route_waypoint[0].transform.location.y, 'y'),
                    #                    0),
                    #     carla.Rotation(0, 0, 0)))

                    if last_waypoint_transform is not None:
                        # Ensuring that the next waypoint is in front of the previous
                        if -1 == is_in_front_of_waypoint_from_vector(line_point=last_waypoint_transform,
                                                                          vector_line=last_waypoint_transform.get_right_vector(),
                                                                          in_front_point=route_waypoint[0].transform):
                            last_waypoint_transform = route_waypoint[0].transform

                            route.append(route_waypoint[0].transform)

                    else:
                        last_waypoint_transform = route_waypoint[0].transform
                        route.append(route_waypoint[0].transform)



                print(f"length of route {len(route)}")
                print(f"Radii: {get_radii(route=route,last_waypoint_index=0,no_of_points_to_calculate_chord=5)}")
                all_routes.append(route)

    plot_all_routes(all_routes=all_routes,all_spawn_points=map.get_spawn_points())



# for _ in range(1000):
#     print(get_entry_exit_map_locations())