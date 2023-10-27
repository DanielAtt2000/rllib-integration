from rllib_integration.GetStartStopLocation import spawn_points_2_lane_roundabout_small_easy, \
    spawn_points_2_lane_roundabout_small_difficult, lower_medium_roundabout_easy, lower_medium_roundabout_difficult, \
    upper_medium_roundabout, roundabout20m



def get_route_type(current_entry_idx, current_exit_idx):
    found = False
    ifroundabout20m = False
    ifupper_medium_roundabout = False
    iflower_medium_roundabout_all = False
    ifdoubleRoundabout_roundabout_all = False
    for entry_easy in roundabout20m:
        entry_idx = entry_easy[0]
        if current_entry_idx == entry_idx:
            if current_exit_idx in entry_easy[1]:
                found = True
                ifroundabout20m = True
                break

    if not found:
        for entry_easy in upper_medium_roundabout:
            entry_idx = entry_easy[0]
            if current_entry_idx == entry_idx:
                if current_exit_idx in entry_easy[1]:
                    found = True
                    ifupper_medium_roundabout = True
                    break
    if not found:
        for entry_easy in (lower_medium_roundabout_easy+lower_medium_roundabout_difficult):
            entry_idx = entry_easy[0]
            if current_entry_idx == entry_idx:
                if current_exit_idx in entry_easy[1]:
                    found = True
                    iflower_medium_roundabout_all = True
                    break
    if not found:
        for entry_easy in (spawn_points_2_lane_roundabout_small_easy+spawn_points_2_lane_roundabout_small_difficult):
            entry_idx = entry_easy[0]
            if current_entry_idx == entry_idx:
                if current_exit_idx in entry_easy[1]:
                    found = True
                    ifdoubleRoundabout_roundabout_all = True
                    break

    if iflower_medium_roundabout_all:
        return '32m'
    elif ifupper_medium_roundabout:
        return '40m'
    elif ifroundabout20m:
        return '20m'
    elif ifdoubleRoundabout_roundabout_all:
        return 'double'
    else:
        raise Exception('fuc')