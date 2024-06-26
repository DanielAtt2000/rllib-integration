import math

import pandas as pd
import random

from Helper import open_pickle
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

lower_medium_roundabout_all = [
    # IN EDITOR     # IN EDITOR
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
    # IN EDITOR     # IN EDITOR

    # # IN EXPORT IN EXPORT
    # [21, [9], "left"],
    # [21, [48], "left"],
    # [21, [64], "left"],
    # [29, [36], "right"],
    # [29, [55], "right"],
    #
    # # [4, [48], "left"],
    # [4, [64], "left"],
    # # [4, [17], "left"],
    # # [5, [55], "right"],
    # [5, [67], "right"],
    #
    # [10, [64], "left"],
    # # [10, [17], "left"],
    # [10, [9], "left"],
    # [66, [67], "right"],
    # [66, [46], "right"],
    #
    # # [35, [17], "left"],
    # [35, [9], "left"],
    # [35, [48], "left"],
    # # [54, [46], "right"],
    # # [54, [36], "right"],
    # # IN EXPORT IN EXPORT

]

lower_medium_roundabout_easy = [
    # IN EXPORT IN EXPORT
    [21, [9], "left"],
    [29, [36], "right"],
    [29, [55], "right"],

    # [4, [48], "left"],
    # [4, [17], "left"],
    # [5, [55], "right"],
    [5, [67], "right"],

    [10, [64], "left"],
    [66, [67], "right"],
    [66, [46], "right"],

    # IN EXPORT IN EXPORT
]

lower_medium_roundabout_difficult = [
    # [21, [9], "left"],
    [21, [48], "left"],
    [21, [64], "left"],
    # [29, [36], "right"],
    # [29, [55], "right"],
    #
    # # [4, [48], "left"],
    [4, [64], "left"],
    # # [4, [17], "left"],
    # # [5, [55], "right"],
    # [5, [67], "right"],
    #
    # [10, [64], "left"],
    # # [10, [17], "left"],
    [10, [9], "left"],
    # [66, [67], "right"],
    # [66, [46], "right"],
    #
    # # [35, [17], "left"],
    [35, [9], "left"],
    [35, [48], "left"],
    ]

upper_medium_roundabout = [
    # INEDITOR INEDITOR
    # [26, [14], "left"],
    # [26, [3], "left"],
    # [26, [63], "left"],
    # [27, [66], "right"],
    # # [27, [2], "right"],
    #
    # [4, [3], "left"],
    # [4, [63], "left"],
    # [4, [14], "left"],
    # [64, [2], "right"],
    # [64, [25], "right"],
    # INEDITOR INEDITOR

    # INEXPORT

    [19, [6], "left"],
    [19, [23], "left"],
    [19, [60], "left"],
    # [20, [63], "right"],
    # [27, [2], "right"],

    [34, [23], "left"],
    [34, [60], "left"],
    [34, [6], "left"],
    [61, [12], "right"],
    # [61, [18], "right"],

    # [, [], "left"],
    # [, [], "left"],
    # [, [], "left"],
    # [, [], "right"],
    # [, [], "right"],

    # INEXPORT

]

oneAndTwoLaneRoundabout1Lane40m = [
    [140,[91],'oneLane'],
    [140,[179],'oneLane'],
    [140,[125],'oneLane'],
    [188,[179],'oneLane'],
    [188,[125],'oneLane'],
    [188,[91],'oneLane'],
    [98,[125],'oneLane'],
    [98,[91],'oneLane'],
    [98,[179],'oneLane'],
]

oneAndTwoLaneRoundabout1Lane30m = [
    [15,[37],'oneLane'],
    [15,[197],'oneLane'],
    [15,[59],'oneLane'],
    [15,[17],'oneLane'],
    [25,[197],'oneLane'],
    [25,[59],'oneLane'],
    [25,[17],'oneLane'],
    [25,[37],'oneLane'],
    [191,[59],'oneLane'],
    [191,[17],'oneLane'],
    [191,[37],'oneLane'],
    [191,[197],'oneLane'],
    [13,[17],'oneLane'],
    [13,[37],'oneLane'],
    [13,[197],'oneLane'],
    [13,[59],'oneLane'],
]

oneAndTwoLaneRoundabout1Lane20m = [
    [197,[155],'oneLane'],
    [197,[202],'oneLane'],
    [197,[191],'oneLane'],
    [40,[202],'oneLane'],
    [40,[191],'oneLane'],
    [40,[155],'oneLane'],
    [132,[191],'oneLane'],
    [132,[155],'oneLane'],
    [132,[202],'oneLane'],
]

oneAndTwoLaneRoundabout1Lane16m = [
    [26,[13],'oneLane'],
    [26,[203],'oneLane'],
    [26,[24],'oneLane'],
    [26,[200],'oneLane'],
    [59,[203],'oneLane'],
    [59,[24],'oneLane'],
    [59,[200],'oneLane'],
    [59,[13],'oneLane'],
    [158,[24],'oneLane'],
    [158,[200],'oneLane'],
    [158,[13],'oneLane'],
    [158,[203],'oneLane'],
    [147,[200],'oneLane'],
    [147,[13],'oneLane'],
    [147,[203],'oneLane'],
    [147,[24],'oneLane'],
]
oneAndTwoLaneRoundabout2Lane16m = [
    [207, [175], 'right'],
    [207, [131], 'right'],
    [174, [160], 'left'],
    [174, [185], 'left'],
    [204, [131], 'right'],
    [204, [186], 'right'],
    [152, [185], 'left'],
    [152, [149], 'left'],
    [192, [186], 'right'],
    [192, [175], 'right'],
    [14, [149], 'left'],
    [14, [160], 'left'],
]

oneAndTwoLaneRoundabout2Lane20m = [
    [0, [38], 'left'],
    [0, [206], 'left'],
    [1, [129], 'right'],
    [1, [148], 'right'],
    [2, [129], 'right'],
    [2, [184], 'right'],
    [52, [38], 'left'],
    [52, [143], 'left'],
    [121, [143], 'left'],
    [121, [206], 'left'],
    [180, [148], 'right'],
    [180, [184], 'right'],
]

oneAndTwoLaneRoundabout2Lane40m = [
    # IN OUTPUT
    # [33, [70], 'left'],
    # [33, [112], 'left'],
    # [33, [137], 'left'],
    # [39, [70], 'left'],
    # [39, [112], 'left'],
    # [39, [136], 'left'],
    # [41, [135], 'right'],
    # [41, [138], 'right'],
    # [55, [47], 'right'],
    # [55, [138], 'right'],
    # [89, [106], 'right'],
    # [89, [135], 'right'],
    # [97, [70], 'left'],
    # [97, [136], 'left'],
    # [97, [137], 'left'],
    # [139, [112], 'left'],
    # [139, [136], 'left'],
    # [139, [137], 'left'],
    # [142, [47], 'right'],
    # [142, [106], 'right'],
    #

# IN EDITOR
#
#     [36, [33], 'left'],
#     [36, [34], 'left'],
#     [36, [198], 'left'],
#     [39, [139], 'right'],
#     [39, [192], 'right'],
#     [126, [16], 'left'],
#     [126, [34], 'left'],
#     [126, [198], 'left'],
#     [131, [16], 'left'],
#     [131, [33], 'left'],
#     [131, [198], 'left'],
#     [133, [32], 'right'],
#     [133, [35], 'right'],
#     [146, [35], 'right'],
#     [146, [139], 'right'],
#     [177, [32], 'right'],
#     [177, [192], 'right'],
#     [184, [16], 'left'],
#     [184, [33], 'left'],
#     [184, [34], 'left'],
#

[1, [11], 'left'],
[1, [19], 'left'],
[10, [8], 'left'],
[10, [11], 'left'],
[12, [5], 'right'],
[12, [22], 'right'],
[18, [20], 'right'],
[18, [22], 'right'],
[21, [8], 'left'],
[21, [19], 'left'],
[29, [5], 'right'],
[29, [20], 'right'],



]
oneAndTwoLaneRoundabout2Lane30m = [
    [51, [64], 'right'],
    [51, [92], 'right'],
    [76, [64], 'right'],
    [76, [182], 'right'],
    [93, [58], 'left'],
    [93, [111], 'left'],
    [117, [85], 'left'],
    [117, [111], 'left'],
    [183, [92], 'right'],
    [183, [182], 'right'],
    [196, [58], 'left'],
    [196, [85], 'left'],
]
# [, [], 'oneLane'],
# [, [], 'oneLane'],

oneAndTwoLaneRoundabouts = oneAndTwoLaneRoundabout2Lane40m

roundabout20m = [

    # INEDITOR INEDOTR

    # [8, [14], "right"],
    # [8, [35], "right"],
    #
    # [32, [2], "left"],
    # [32, [31], "left"],
    # [32, [4], "left"],
    # [24, [35], "right"],
    # [24, [6], "right"],
    #
    # [20, [31], "left"],
    # # [20, [4], "left"],
    # [20, [7], "left"],
    # [19, [6], "right"],
    # [19, [26], "right"],
    #
    # [1, [4], "left"],
    # [1, [7], "left"],
    # [1, [2], "left"],
    # [0, [26], "right"],
    # [0, [14], "right"],
    # # INEDITOR INEDOTR

    # IN EXPORT
    [26, [12], "left"],
    [26, [25], "left"],
    [26, [30], "left"],
    [17, [29], "right"],
    [17, [32], "right"],

    [13, [25], "left"],
    # [13, [30], "left"],
    [13, [33], "left"],
    [11, [32], "right"],
    [11, [19], "right"],

    [1, [30], "left"],
    [1, [33], "left"],
    [1, [12], "left"],
    [0, [19], "right"],
    [0, [6], "right"],

    [34, [33], "left"],
    # [34, [12], "left"],
    [34, [25], "left"],
    # [, [], "right"],
    # [, [], "right"],

    # IN EXPORT

]

oneLane50m = [
    [1, [0], 'oneLane'],
    [1, [2], 'oneLane'],
    [1, [4], 'oneLane'],
    [1, [6], 'oneLane'],
    [3, [0], 'oneLane'],
    [3, [2], 'oneLane'],
    [3, [4], 'oneLane'],
    [3, [6], 'oneLane'],
    [5, [0], 'oneLane'],
    [5, [2], 'oneLane'],
    [5, [4], 'oneLane'],
    [5, [6], 'oneLane'],
    [7, [0], 'oneLane'],
    [7, [2], 'oneLane'],
    [7, [4], 'oneLane'],
    [7, [6], 'oneLane'],
]

oneLane70m = [
    [8, [9], 'oneLane'],
    [8, [10], 'oneLane'],
    [8, [12], 'oneLane'],
    [8, [14], 'oneLane'],
    [11, [9], 'oneLane'],
    [11, [10], 'oneLane'],
    [11, [12], 'oneLane'],
    [11, [14], 'oneLane'],
    [13, [9], 'oneLane'],
    [13, [10], 'oneLane'],
    [13, [12], 'oneLane'],
    [13, [14], 'oneLane'],
    [15, [9], 'oneLane'],
    [15, [10], 'oneLane'],
    [15, [12], 'oneLane'],
    [15, [14], 'oneLane'],
]

OneLaneRoundabouts = oneLane50m + oneLane70m




# roundabouts = [spawn_points_2_lane_roundabout_difficult,spawn_points_2_lane_roundabout_easy]

def get_entry_spawn_points(map_name):
    entry_spawn_points = set()
    if map_name == 'mediumRoundabout4':
        # total of 13 routes lower
        roundabouts = [lower_medium_roundabout_easy,lower_medium_roundabout_difficult]
        # total of 7 routes lower
        # roundabouts = [upper_medium_roundabout]
    elif map_name == 'doubleRoundabout37':
        # total of 13 routes
        roundabouts = [spawn_points_2_lane_roundabout_small_easy,spawn_points_2_lane_roundabout_small_difficult]
    elif map_name == '20m':
        # total of 17 routes
        roundabouts = [roundabout20m]
    elif map_name == 'OneLaneRoundabouts':
        # total of 17 routes
        roundabouts = [OneLaneRoundabouts]
    else:
        raise Exception('Roundabout name not entered')

    for roundabout in roundabouts:
        for route in roundabout:
            entry_spawn_points.add(route[0])

    return entry_spawn_points


def get_entry_exit_spawn_point_indices_2_lane(failed_spawn_locations, last_roundabout_choice, last_chosen_route, map_name, is_testing):
    if map_name == 'mediumRoundabout4':

        mediumRoundaboutType = open_pickle('mediumRoundabout4Type')

        if mediumRoundaboutType == 'training' or mediumRoundaboutType == '':
            # total of 13 routes lower
            roundabouts = [lower_medium_roundabout_easy+lower_medium_roundabout_difficult]
            # roundabouts = [lower_medium_roundabout_difficult]
        elif mediumRoundaboutType == 'testing':
            # total of 7 routes lower
            roundabouts = [upper_medium_roundabout]
        else:
            raise Exception('Error with mediumRoundaboutType')

    elif map_name == 'doubleRoundabout37':
        # total of 13 routes
        roundabouts = [spawn_points_2_lane_roundabout_small_easy+spawn_points_2_lane_roundabout_small_difficult]
    elif map_name == '20m':
        # total of 17 routes
        roundabouts = [roundabout20m]
    elif map_name == 'OneLaneRoundabouts':
        # total of 17 routes
        roundabouts = [OneLaneRoundabouts]
    if is_testing:
        assert len(roundabouts) == 1

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
        if last_chosen_route != -2 :
            element = roundabout_choice[last_chosen_route]

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

    return entry_spawn_point_index, exit_spawn_point_index, route_lane, last_roundabout_choice, len(roundabouts[0])

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

def visualise_all_routes(map,map_name):
    all_routes = []

    x = print('----------------------------\n\n\n'
              'Ensure that the roundabouts have been updated in here as well'
              '-----------------------------\n\n\n')

    if map_name == 'mediumRoundabout4':
        roundabouts = [upper_medium_roundabout]
    elif map_name == 'doubleRoundabout37':
        roundabouts = [spawn_points_2_lane_roundabout_small_easy,spawn_points_2_lane_roundabout_small_difficult]
    elif map_name == '20m':
        # total of 17 routes
        roundabouts = [roundabout20m]
    elif map_name == 'oneAndTwoLaneRoundabouts':
        roundabouts = [oneAndTwoLaneRoundabouts]
    elif map_name == 'OneLaneRoundabouts':
        # total of 17 routes
        roundabouts = [OneLaneRoundabouts]

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

    visualise_custom_route_among_others = False
    if visualise_custom_route_among_others:
        class Transform():
            def __init__(self, x, y):
                self.location = Location(x, y)

        class Location():
            def __init__(self, x, y):
                self.x = x
                self.y = y
        temp_route = []
        if visualise_custom_route_among_others:
            temp_route_points = [(1.9168671369552612, 44.50431442260742), (1.9168671369552612, 44.50431442260742), (1.9168992042541504, 44.499549865722656), (1.9175279140472412, 44.49787902832031), (1.918110966682434, 44.49878692626953), (1.917875051498413, 44.49916076660156), (1.9173415899276733, 44.499324798583984), (1.916885495185852, 44.49937438964844), (1.9166066646575928, 44.499324798583984), (1.916442632675171, 44.499183654785156), (1.9163579940795898, 44.49898910522461), (1.916322946548462, 44.49877166748047), (1.9162724018096924, 44.49855422973633), (1.9162400960922241, 44.49835968017578), (1.916219711303711, 44.49821090698242), (1.9162076711654663, 44.49811935424805), (1.9162009954452515, 44.49806213378906), (1.9161977767944336, 44.49802017211914), (1.9161967039108276, 44.49800109863281), (1.9161961078643799, 44.49799346923828), (1.9161961078643799, 44.497982025146484), (1.9161968231201172, 44.497982025146484), (1.9161978960037231, 44.49798583984375), (1.9161988496780396, 44.49799346923828), (1.91864013671875, 44.48957061767578), (1.9330692291259766, 44.45561599731445), (1.9662450551986694, 44.40485382080078), (2.0155534744262695, 44.33753967285156), (2.0655839443206787, 44.24091339111328), (2.0905230045318604, 44.10956954956055), (2.0780041217803955, 43.95721435546875), (2.0376386642456055, 43.79319763183594), (1.987322449684143, 43.61381912231445), (1.9295530319213867, 43.42069625854492), (1.8924040794372559, 43.208621978759766), (1.9061851501464844, 42.98324966430664), (1.9620660543441772, 42.757015228271484), (2.0060386657714844, 42.518272399902344), (2.0102696418762207, 42.266563415527344), (2.014005422592163, 42.00710678100586), (2.0193207263946533, 41.746986389160156), (2.0250566005706787, 41.48699188232422), (2.0308446884155273, 41.22739028930664), (2.036614418029785, 40.96831130981445), (2.042405366897583, 40.7075080871582), (2.048455238342285, 40.434146881103516), (2.075159788131714, 40.150569915771484), (2.1552772521972656, 39.873104095458984), (2.2444207668304443, 39.589317321777344), (2.2758944034576416, 39.28450012207031), (2.2789556980133057, 38.9685173034668), (2.2287139892578125, 38.65558624267578), (2.1859238147735596, 38.338523864746094), (2.215047836303711, 38.02394104003906), (2.3040056228637695, 37.729515075683594), (2.3986918926239014, 37.4418830871582), (2.413377285003662, 37.14981460571289), (2.3668246269226074, 36.86860275268555), (2.296964168548584, 36.59102249145508), (2.219513416290283, 36.312862396240234), (2.1678295135498047, 36.028263092041016), (2.1820316314697266, 35.743927001953125), (2.2493896484375, 35.474822998046875), (2.3639984130859375, 35.23506164550781), (2.48650860786438, 35.00186538696289), (2.5388503074645996, 34.745513916015625), (2.543104410171509, 34.487300872802734), (2.5044238567352295, 34.2415771484375), (2.447556495666504, 34.00016403198242), (2.4163594245910645, 33.749935150146484), (2.426269054412842, 33.49304962158203), (2.4364500045776367, 33.23011779785156), (2.4449827671051025, 32.96294021606445), (2.4531047344207764, 32.69431686401367), (2.46113920211792, 32.427127838134766), (2.469090461730957, 32.16340637207031), (2.4956600666046143, 31.90761947631836), (2.5253686904907227, 31.659547805786133), (2.535818338394165, 31.4123592376709), (2.546659231185913, 31.158573150634766), (2.55786395072937, 30.90963363647461), (2.569291830062866, 30.65895652770996), (2.5807619094848633, 30.407880783081055), (2.592228651046753, 30.15679931640625), (2.6036858558654785, 29.905824661254883), (2.6334898471832275, 29.658926010131836), (2.6661486625671387, 29.416242599487305), (2.6802380084991455, 29.16988754272461), (2.6946518421173096, 28.9208927154541), (2.7096734046936035, 28.671030044555664), (2.724785089492798, 28.422008514404297), (2.739792823791504, 28.174991607666016), (2.7726149559020996, 27.934545516967773), (2.8478481769561768, 27.71417236328125), (2.9614150524139404, 27.52344512939453), (3.0793800354003906, 27.334579467773438), (3.133193254470825, 27.115468978881836), (3.1281588077545166, 26.888856887817383), (3.093200206756592, 26.660383224487305), (3.0738279819488525, 26.42222785949707), (3.0888912677764893, 26.17990493774414), (3.135328531265259, 25.941423416137695), (3.222684860229492, 25.721446990966797), (3.3096046447753906, 25.504547119140625), (3.3477611541748047, 25.276641845703125), (3.3825972080230713, 25.04860496520996), (3.4190456867218018, 24.821134567260742), (3.45530366897583, 24.597753524780273), (3.490992307662964, 24.37834930419922), (3.5435404777526855, 24.16267967224121), (3.5982203483581543, 23.945377349853516), (3.637427568435669, 23.718358993530273), (3.678067207336426, 23.484867095947266), (3.720006227493286, 23.246707916259766), (3.762420892715454, 23.006515502929688), (3.8220772743225098, 22.772966384887695), (3.883173704147339, 22.545557022094727), (3.9261975288391113, 22.31654167175293), (3.9693686962127686, 22.087921142578125), (4.0131754875183105, 21.85854148864746), (4.057350158691406, 21.627864837646484), (4.101762294769287, 21.39603042602539), (4.146286964416504, 21.163578033447266), (4.207763671875, 20.937294006347656), (4.270707130432129, 20.715139389038086), (4.316457748413086, 20.489126205444336), (4.362425327301025, 20.2632999420166), (4.4087934494018555, 20.037853240966797), (4.471762180328369, 19.81897735595703), (4.572335243225098, 19.622936248779297), (4.67173957824707, 19.42936897277832), (4.7116618156433105, 19.215736389160156), (4.751945495605469, 18.99593734741211), (4.808287143707275, 18.770978927612305), (4.883586883544922, 18.545318603515625), (4.999228000640869, 18.33978843688965), (5.115942478179932, 18.135164260864258), (5.169525623321533, 17.910200119018555), (5.197907447814941, 17.679899215698242), (5.253490924835205, 17.4547176361084), (5.358188152313232, 17.25273895263672), (5.498903751373291, 17.08095359802246), (5.64558219909668, 16.91006851196289), (5.746681213378906, 16.70366096496582), (5.7908782958984375, 16.47897720336914), (5.789604187011719, 16.25285530090332), (5.789767265319824, 16.025724411010742), (5.849433898925781, 15.807478904724121), (5.956360340118408, 15.61384391784668), (6.094935894012451, 15.451208114624023), (6.235894203186035, 15.291685104370117), (6.330788612365723, 15.102460861206055), (6.3733439445495605, 14.895966529846191), (6.419039249420166, 14.683333396911621), (6.498636722564697, 14.471048355102539), (6.581948280334473, 14.250054359436035), (6.68304967880249, 14.031007766723633), (6.820671081542969, 13.835969924926758), (6.973982810974121, 13.652463912963867), (7.105587482452393, 13.453866958618164), (7.1741838455200195, 13.233501434326172), (7.2021684646606445, 13.0078706741333), (7.252418041229248, 12.784286499023438), (7.362324237823486, 12.586795806884766), (7.50571870803833, 12.411824226379395), (7.633763313293457, 12.220595359802246), (7.7009782791137695, 12.005545616149902), (7.766129970550537, 11.793563842773438), (7.881801605224609, 11.613334655761719), (8.022307395935059, 11.456809043884277), (8.167555809020996, 11.303723335266113), (8.295296669006348, 11.131999015808105), (8.381902694702148, 10.928062438964844), (8.43182373046875, 10.705239295959473), (8.467328071594238, 10.47010326385498), (8.530866622924805, 10.233869552612305), (8.655804634094238, 10.026493072509766), (8.813834190368652, 9.846833229064941), (8.979823112487793, 9.673203468322754), (9.149078369140625, 9.503543853759766), (9.321882247924805, 9.3392972946167), (9.475092887878418, 9.158914566040039), (9.56966495513916, 8.94925594329834), (9.625103950500488, 8.729192733764648), (9.672686576843262, 8.50406265258789), (9.747063636779785, 8.283228874206543), (9.878462791442871, 8.093361854553223), (10.04081916809082, 7.929904460906982), (10.211174964904785, 7.771634578704834), (10.361612319946289, 7.5954203605651855), (10.454168319702148, 7.39013147354126), (10.508163452148438, 7.175703048706055), (10.554129600524902, 6.958712100982666), (10.595316886901855, 6.742944717407227), (10.661153793334961, 6.53380823135376), (10.782293319702148, 6.351047039031982), (10.93564510345459, 6.189268589019775), (11.101045608520508, 6.027708053588867), (11.27431583404541, 5.86537504196167), (11.454487800598145, 5.705340385437012), (11.617024421691895, 5.526950359344482), (11.720888137817383, 5.317055702209473), (11.776857376098633, 5.097208499908447), (11.798860549926758, 4.875341415405273), (11.842620849609375, 4.651327610015869), (11.953274726867676, 4.451389789581299), (12.10507869720459, 4.28126335144043), (12.27375316619873, 4.123032093048096), (12.423978805541992, 3.9451539516448975), (12.515645027160645, 3.738173246383667), (12.560803413391113, 3.5230417251586914), (12.57298469543457, 3.3081541061401367), (12.576403617858887, 3.0926454067230225), (12.6032075881958, 2.874662399291992), (12.697627067565918, 2.670795440673828), (12.836440086364746, 2.4890856742858887), (12.995051383972168, 2.3155624866485596), (13.160941123962402, 2.1413493156433105), (13.307883262634277, 1.9486156702041626), (13.394014358520508, 1.7280770540237427), (13.431187629699707, 1.501075267791748), (13.433674812316895, 1.2754993438720703), (13.426488876342773, 1.048311471939087), (13.411333084106445, 0.8199331760406494), (13.418305397033691, 0.5864549279212952), (13.495336532592773, 0.36649924516677856), (13.617801666259766, 0.1709594875574112), (13.757858276367188, -0.013907451182603836), (13.902541160583496, -0.19720390439033508), (14.025209426879883, -0.3946765959262848), (14.086714744567871, -0.6123936772346497), (14.101323127746582, -0.831364095211029), (14.0714111328125, -1.038615345954895), (14.032526969909668, -1.2439677715301514), (13.998353004455566, -1.4512734413146973), (13.98569393157959, -1.6682215929031372), (14.041126251220703, -1.884535312652588), (14.143987655639648, -2.087691068649292), (14.239086151123047, -2.3008546829223633), (14.2703857421875, -2.5309953689575195), (14.252599716186523, -2.7600200176239014), (14.200509071350098, -2.9826364517211914), (14.125215530395508, -3.196154832839966), (14.050263404846191, -3.412649393081665), (14.046900749206543, -3.644218921661377), (14.106599807739258, -3.865863561630249), (14.164666175842285, -4.088497638702393), (14.158072471618652, -4.313914775848389), (14.104545593261719, -4.529069423675537), (14.022247314453125, -4.731567859649658), (13.921415328979492, -4.920016765594482), (13.822463989257812, -5.109558582305908), (13.789626121520996, -5.321120262145996), (13.815377235412598, -5.530513763427734), (13.87142276763916, -5.739749431610107), (13.90532112121582, -5.96515417098999), (13.884268760681152, -6.199465274810791), (13.818290710449219, -6.427695274353027), (13.713869094848633, -6.638551712036133), (13.593459129333496, -6.84373664855957), (13.456154823303223, -7.031868934631348), (13.318408012390137, -7.219982624053955), (13.245771408081055, -7.441908836364746), (13.237394332885742, -7.671529769897461), (13.261919021606445, -7.899984359741211), (13.260933876037598, -8.133077621459961), (13.195395469665527, -8.35401439666748), (13.08676528930664, -8.553677558898926), (12.949864387512207, -8.73388671875), (12.802546501159668, -8.91140365600586), (12.641485214233398, -9.070252418518066), (12.480643272399902, -9.228899002075195), (12.377713203430176, -9.430068016052246), (12.33506965637207, -9.647425651550293), (12.323862075805664, -9.866657257080078), (12.288474082946777, -10.08287239074707), (12.20317268371582, -10.280915260314941), (12.081292152404785, -10.458968162536621), (11.927515983581543, -10.609210014343262), (11.74855899810791, -10.729019165039062), (11.56772232055664, -10.869333267211914), (11.437381744384766, -11.065937042236328), (11.365595817565918, -11.291247367858887), (11.29216480255127, -11.51740837097168), (11.16191291809082, -11.708959579467773), (10.997737884521484, -11.864058494567871), (10.81119155883789, -11.97835922241211), (10.619833946228027, -12.09577465057373), (10.472869873046875, -12.271366119384766), (10.379953384399414, -12.480931282043457), (10.317459106445312, -12.703553199768066), (10.230509757995605, -12.919013023376465), (10.09031867980957, -13.09638500213623), (9.921557426452637, -13.23593807220459), (9.736390113830566, -13.33255386352539), (9.54934024810791, -13.429643630981445), (9.395920753479004, -13.586976051330566), (9.255926132202148, -13.770928382873535), (9.095076560974121, -13.948034286499023), (8.903616905212402, -14.08955192565918), (8.693340301513672, -14.185697555541992), (8.47883129119873, -14.27634334564209), (8.302765846252441, -14.428318977355957), (8.175410270690918, -14.618889808654785), (8.04593563079834, -14.809683799743652), (7.870701789855957, -14.955202102661133), (7.670892238616943, -15.058672904968262), (7.459194183349609, -15.116670608520508), (7.2406840324401855, -15.143328666687012), (7.014665603637695, -15.190706253051758), (6.811807632446289, -15.310138702392578), (6.650818347930908, -15.479262351989746), (6.516581058502197, -15.668317794799805), (6.361899375915527, -15.841841697692871), (6.170090198516846, -15.961432456970215), (5.9642863273620605, -16.035734176635742), (5.758316993713379, -16.063955307006836), (5.554304122924805, -16.06229019165039), (5.3413519859313965, -16.08087158203125), (5.13862419128418, -16.171340942382812), (4.963870048522949, -16.316226959228516), (4.782912254333496, -16.464521408081055), (4.570773124694824, -16.570114135742188), (4.344139099121094, -16.63163948059082), (4.116827964782715, -16.64621353149414), (3.8970272541046143, -16.618608474731445), (3.667827606201172, -16.604921340942383), (3.444179058074951, -16.666271209716797), (3.247718095779419, -16.786012649536133), (3.0696191787719727, -16.933008193969727), (2.8737144470214844, -17.0611515045166), (2.653834819793701, -17.128013610839844), (2.431870222091675, -17.146738052368164), (2.221165657043457, -17.120023727416992), (2.0096869468688965, -17.092084884643555), (1.800780177116394, -17.13397979736328), (1.6143953800201416, -17.23044204711914), (1.43831467628479, -17.355335235595703), (1.2389968633651733, -17.464473724365234), (1.0125491619110107, -17.515012741088867), (0.7792404294013977, -17.517847061157227), (0.5474265217781067, -17.485151290893555), (0.31290313601493835, -17.441162109375), (0.08858919143676758, -17.375442504882812), (-0.1360437572002411, -17.308744430541992), (-0.37056028842926025, -17.31476593017578), (-0.5901036858558655, -17.38433074951172), (-0.797004759311676, -17.484403610229492), (-1.016700029373169, -17.561899185180664), (-1.2465744018554688, -17.57455062866211), (-1.470789909362793, -17.539506912231445), (-1.676609754562378, -17.46104621887207), (-1.883239984512329, -17.37065315246582), (-2.0975558757781982, -17.282291412353516), (-2.297877073287964, -17.174793243408203), (-2.49798846244812, -17.066326141357422), (-2.722288131713867, -17.02560043334961), (-2.9444854259490967, -17.04806900024414), (-3.1592257022857666, -17.101505279541016), (-3.3788976669311523, -17.130889892578125), (-3.5941832065582275, -17.106962203979492), (-3.798983097076416, -17.04246711730957), (-3.9857563972473145, -16.939931869506836), (-4.150937557220459, -16.804813385009766), (-4.336742877960205, -16.673582077026367), (-4.562031269073486, -16.60655975341797), (-4.7981438636779785, -16.60377311706543), (-5.036706447601318, -16.600780487060547), (-5.266725540161133, -16.54473304748535), (-5.476146221160889, -16.44671058654785), (-5.655040740966797, -16.31369400024414), (-5.8324480056762695, -16.17525291442871), (-6.045857906341553, -16.09821891784668), (-6.271524906158447, -16.08036994934082), (-6.500678062438965, -16.061044692993164), (-6.716671466827393, -15.988905906677246), (-6.927902698516846, -15.914172172546387), (-7.136751174926758, -15.841733932495117), (-7.321051597595215, -15.734598159790039), (-7.471728801727295, -15.598238945007324), (-7.593831539154053, -15.437727928161621), (-7.739082336425781, -15.27493953704834), (-7.939594745635986, -15.15894603729248), (-8.168387413024902, -15.08201789855957), (-8.402685165405273, -15.003165245056152), (-8.622121810913086, -14.888211250305176), (-8.809819221496582, -14.737814903259277), (-8.955636024475098, -14.56109619140625), (-9.095733642578125, -14.38129997253418), (-9.283584594726562, -14.254627227783203), (-9.496172904968262, -14.18317985534668), (-9.7117919921875, -14.109195709228516), (-9.90575885772705, -13.985107421875), (-10.071077346801758, -13.825644493103027), (-10.198797225952148, -13.640438079833984), (-10.288018226623535, -13.438596725463867), (-10.394829750061035, -13.234567642211914), (-10.564105033874512, -13.07298469543457), (-10.769434928894043, -12.965349197387695), (-10.98646354675293, -12.888734817504883), (-11.194782257080078, -12.788713455200195), (-11.370319366455078, -12.64382553100586), (-11.510621070861816, -12.472376823425293), (-11.614249229431152, -12.28565788269043), (-11.690115928649902, -12.092337608337402), (-11.771211624145508, -11.894384384155273), (-11.915776252746582, -11.72390079498291), (-12.087310791015625, -11.565773963928223), (-12.248404502868652, -11.389083862304688), (-12.370038032531738, -11.18691349029541), (-12.489030838012695, -10.984090805053711), (-12.656855583190918, -10.827239990234375), (-12.82913875579834, -10.675708770751953), (-13.000267028808594, -10.522780418395996), (-13.169572830200195, -10.368688583374023), (-13.297289848327637, -10.180830001831055), (-13.382007598876953, -9.973705291748047), (-13.420645713806152, -9.758611679077148), (-13.46690845489502, -9.536893844604492), (-13.58198070526123, -9.34158706665039), (-13.743566513061523, -9.188931465148926), (-13.923809051513672, -9.061750411987305), (-14.107763290405273, -8.938440322875977), (-14.276022911071777, -8.793766021728516), (-14.39879322052002, -8.606420516967773), (-14.481574058532715, -8.39622974395752), (-14.5325345993042, -8.173359870910645), (-14.574910163879395, -7.941735744476318), (-14.609472274780273, -7.70504903793335), (-14.634561538696289, -7.465939044952393), (-14.681011199951172, -7.227720260620117), (-14.794422149658203, -7.019495487213135), (-14.94760799407959, -6.846972942352295), (-15.114420890808105, -6.689518451690674), (-15.26191234588623, -6.512879371643066), (-15.352936744689941, -6.305174350738525), (-15.398399353027344, -6.085969924926758), (-15.410429000854492, -5.86263370513916), (-15.413031578063965, -5.633637428283691), (-15.407611846923828, -5.401103496551514), (-15.3929443359375, -5.168119430541992), (-15.368846893310547, -4.9365339279174805), (-15.3658447265625, -4.701244354248047), (-15.433267593383789, -4.479742050170898), (-15.546278953552246, -4.282992362976074), (-15.67470932006836, -4.099201679229736), (-15.805109977722168, -3.919523000717163), (-15.91427230834961, -3.724612236022949), (-15.965423583984375, -3.5076048374176025), (-15.971117973327637, -3.2854363918304443), (-15.943318367004395, -3.063688278198242), (-15.905040740966797, -2.837229013442993), (-15.85811996459961, -2.6086220741271973), (-15.801849365234375, -2.3805859088897705), (-15.736321449279785, -2.1553990840911865), (-15.691070556640625, -1.92412531375885), (-15.717569351196289, -1.6959378719329834), (-15.792702674865723, -1.484785556793213), (-15.886028289794922, -1.280508279800415), (-15.984312057495117, -1.0753577947616577), (-16.05926513671875, -0.8587066531181335), (-16.07127571105957, -0.6354399919509888), (-16.037919998168945, -0.4214189946651459), (-15.974214553833008, -0.21948395669460297), (-15.902267456054688, -0.020007800310850143), (-15.823686599731445, 0.178120955824852), (-15.73542308807373, 0.3798278868198395), (-15.636110305786133, 0.5850893259048462), (-15.525674819946289, 0.7919259667396545), (-15.431574821472168, 1.0141445398330688), (-15.40895938873291, 1.2517335414886475), (-15.439764976501465, 1.4831877946853638), (-15.459566116333008, 1.7141976356506348), (-15.414376258850098, 1.9351164102554321), (-15.325214385986328, 2.1369595527648926), (-15.206844329833984, 2.3227646350860596), (-15.077463150024414, 2.50839900970459), (-14.939173698425293, 2.693134069442749), (-14.791946411132812, 2.8747079372406006), (-14.659208297729492, 3.072551727294922), (-14.593478202819824, 3.297863721847534), (-14.58032512664795, 3.5275206565856934), (-14.556222915649414, 3.755276918411255), (-14.471158027648926, 3.96056866645813), (-14.35287857055664, 4.142805576324463), (-14.219074249267578, 4.307960033416748), (-14.07271957397461, 4.453219890594482), (-13.930712699890137, 4.612517833709717), (-13.781098365783691, 4.769683837890625), (-13.629523277282715, 4.934756755828857), (-13.532307624816895, 5.141866207122803), (-13.48328685760498, 5.367604732513428), (-13.422607421875, 5.595235347747803), (-13.301945686340332, 5.794381141662598), (-13.144775390625, 5.960134983062744), (-12.984583854675293, 6.124788761138916), (-12.848917961120605, 6.313446998596191), (-12.697049140930176, 6.48706579208374), (-12.545351028442383, 6.659330368041992), (-12.450014114379883, 6.8641357421875), (-12.356204986572266, 7.070706844329834), (-12.21706485748291, 7.241691589355469), (-12.051393508911133, 7.374176025390625), (-11.872747421264648, 7.482676029205322), (-11.70388412475586, 7.610593318939209), (-11.585610389709473, 7.782649517059326), (-11.507466316223145, 7.979946136474609), (-11.468976974487305, 8.19216251373291), (-11.431676864624023, 8.415984153747559), (-11.329718589782715, 8.632770538330078), (-11.173501968383789, 8.818243980407715), (-10.985992431640625, 8.967414855957031), (-10.782188415527344, 9.090449333190918), (-10.57258415222168, 9.205836296081543), (-10.35949420928955, 9.312941551208496), (-10.155670166015625, 9.439528465270996), (-10.002870559692383, 9.618051528930664), (-9.89730453491211, 9.821915626525879), (-9.810885429382324, 10.034454345703125), (-9.697288513183594, 10.2368745803833), (-9.533534049987793, 10.39806842803955), (-9.342384338378906, 10.52033805847168), (-9.147817611694336, 10.642086029052734), (-8.998098373413086, 10.815666198730469), (-8.891610145568848, 11.018003463745117), (-8.802148818969727, 11.232718467712402), (-8.686540603637695, 11.438949584960938), (-8.521799087524414, 11.60141372680664), (-8.33278751373291, 11.722007751464844), (-8.14410400390625, 11.83997917175293), (-8.002135276794434, 12.005538940429688), (-7.904185771942139, 12.194546699523926), (-7.848596096038818, 12.391121864318848), (-7.796032428741455, 12.592202186584473), (-7.694281578063965, 12.786307334899902), (-7.5389251708984375, 12.94895076751709), (-7.351633071899414, 13.079084396362305), (-7.1581902503967285, 13.211227416992188), (-7.010723114013672, 13.395408630371094), (-6.9096808433532715, 13.60724925994873), (-6.854738712310791, 13.829703330993652), (-6.844681739807129, 14.051911354064941), (-6.819977283477783, 14.280322074890137), (-6.721866130828857, 14.491904258728027), (-6.5709075927734375, 14.667909622192383), (-6.391157150268555, 14.80904483795166), (-6.2111430168151855, 14.953436851501465), (-6.084787368774414, 15.143720626831055), (-6.007412433624268, 15.353103637695312), (-5.974834442138672, 15.565879821777344), (-5.983969688415527, 15.772174835205078), (-6.008205413818359, 15.97652530670166), (-6.011335372924805, 16.18851089477539), (-5.941963195800781, 16.399425506591797), (-5.81557559967041, 16.590383529663086), (-5.652849197387695, 16.755523681640625), (-5.487387180328369, 16.92674446105957), (-5.3803324699401855, 17.14072036743164), (-5.325718402862549, 17.37148666381836), (-5.3187761306762695, 17.60219383239746), (-5.35598087310791, 17.822582244873047), (-5.380649089813232, 18.051816940307617), (-5.346381664276123, 18.286802291870117), (-5.277092933654785, 18.51578140258789), (-5.202790260314941, 18.741291046142578), (-5.166833400726318, 18.97330093383789), (-5.136120319366455, 19.207977294921875), (-5.107719421386719, 19.44409942626953), (-5.053944110870361, 19.672767639160156), (-4.952328205108643, 19.876476287841797), (-4.827730178833008, 20.063505172729492), (-4.7255072593688965, 20.263301849365234), (-4.684974193572998, 20.476863861083984), (-4.6896514892578125, 20.685951232910156), (-4.736240863800049, 20.884742736816406), (-4.784289836883545, 21.092763900756836), (-4.762205600738525, 21.315343856811523), (-4.679058074951172, 21.528581619262695), (-4.563558101654053, 21.732999801635742), (-4.439136981964111, 21.93992805480957), (-4.33762264251709, 22.16132926940918), (-4.301991939544678, 22.39622688293457), (-4.315812110900879, 22.6259708404541), (-4.374699115753174, 22.838136672973633), (-4.462902545928955, 23.03750228881836), (-4.536098957061768, 23.252485275268555), (-4.536191940307617, 23.48419952392578), (-4.473574638366699, 23.70693016052246), (-4.37112283706665, 23.90837287902832), (-4.262642860412598, 24.107179641723633), (-4.1594696044921875, 24.309871673583984), (-4.08005952835083, 24.52164077758789), (-4.0624237060546875, 24.742319107055664), (-4.090266704559326, 24.961376190185547), (-4.161820411682129, 25.167341232299805), (-4.263576030731201, 25.363088607788086), (-4.352725505828857, 25.578012466430664), (-4.369155406951904, 25.81581687927246), (-4.3205108642578125, 26.048458099365234), (-4.229023456573486, 26.26317024230957), (-4.130379676818848, 26.47623634338379), (-4.036963939666748, 26.69386100769043), (-3.967839241027832, 26.92017936706543), (-3.963430166244507, 27.15213966369629), (-4.006115436553955, 27.374826431274414), (-4.0507636070251465, 27.597875595092773), (-4.027495384216309, 27.82343864440918), (-3.954525947570801, 28.037668228149414), (-3.860905647277832, 28.248519897460938), (-3.7611336708068848, 28.46179962158203), (-3.685307264328003, 28.685861587524414), (-3.6746432781219482, 28.916366577148438), (-3.711050271987915, 29.137386322021484), (-3.7590560913085938, 29.3554630279541), (-3.771812915802002, 29.57857894897461), (-3.7232275009155273, 29.789052963256836), (-3.6401524543762207, 29.98497200012207), (-3.576314687728882, 30.19366455078125), (-3.573911666870117, 30.41428565979004), (-3.617467164993286, 30.632808685302734), (-3.704387664794922, 30.835338592529297), (-3.7954955101013184, 31.044261932373047), (-3.8168578147888184, 31.276857376098633), (-3.7738378047943115, 31.506065368652344), (-3.697941780090332, 31.72783660888672), (-3.615093946456909, 31.95030403137207), (-3.5578670501708984, 32.18085479736328), (-3.5664992332458496, 32.4124870300293), (-3.5762312412261963, 32.64070129394531), (-3.525689125061035, 32.856712341308594), (-3.44144868850708, 33.06052017211914), (-3.3797929286956787, 33.27237319946289), (-3.3807082176208496, 33.48712921142578), (-3.424984931945801, 33.69084930419922), (-3.470289945602417, 33.89338302612305), (-3.488906145095825, 34.10376739501953), (-3.510740041732788, 34.32352066040039), (-3.5081465244293213, 34.5538444519043), (-3.4528660774230957, 34.781734466552734), (-3.367969512939453, 35.006103515625), (-3.3066248893737793, 35.24319076538086), (-3.310364246368408, 35.484737396240234), (-3.3301265239715576, 35.72657775878906), (-3.350996732711792, 35.968509674072266), (-3.3745064735412598, 36.20920944213867), (-3.4013564586639404, 36.448238372802734), (-3.401949167251587, 36.68487548828125), (-3.3500120639801025, 36.908111572265625), (-3.271287202835083, 37.12144470214844), (-3.216792106628418, 37.34444808959961), (-3.223830223083496, 37.57243347167969), (-3.245521068572998, 37.804283142089844), (-3.2686550617218018, 38.0405387878418), (-3.2946765422821045, 38.278419494628906), (-3.2943387031555176, 38.515045166015625), (-3.241469144821167, 38.738548278808594), (-3.1620867252349854, 38.95109939575195), (-3.1074392795562744, 39.17092514038086), (-3.1166391372680664, 39.38959884643555), (-3.141129732131958, 39.6052360534668), (-3.1622040271759033, 39.82585525512695), (-3.186336040496826, 40.05264663696289), (-3.2144548892974854, 40.28565216064453)]
            for point in temp_route_points:
                temp_route.append(Transform(point[0], point[1]))
        all_routes.append(temp_route)
    plot_all_routes(all_routes=all_routes,all_spawn_points=map.get_spawn_points())



# for _ in range(1000):
#     print(get_entry_exit_map_locations())