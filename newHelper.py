from rllib_integration.GetStartStopLocation import spawn_points_2_lane_roundabout_small_easy, \
    spawn_points_2_lane_roundabout_small_difficult, lower_medium_roundabout_easy, lower_medium_roundabout_difficult, \
    upper_medium_roundabout, roundabout20m


def get_route_type(current_entry_idx, current_exit_idx):
    lane = ''
    found = False
    ifroundabout20m = False
    ifupper_medium_roundabout = False
    iflower_medium_roundabout_all = False
    if50Roundabout_roundabout_all = False
    if16Roundabout_roundabout_all = False
    for entry_easy in roundabout20m:
        entry_idx = entry_easy[0]
        if current_entry_idx == entry_idx:
            if current_exit_idx in entry_easy[1]:
                found = True
                ifroundabout20m = True
                lane = entry_easy[2]
                break

    if not found:
        for entry_easy in upper_medium_roundabout:
            entry_idx = entry_easy[0]
            if current_entry_idx == entry_idx:
                if current_exit_idx in entry_easy[1]:
                    found = True
                    ifupper_medium_roundabout = True
                    lane = entry_easy[2]
                    break
    if not found:
        for entry_easy in (lower_medium_roundabout_easy+lower_medium_roundabout_difficult):
            entry_idx = entry_easy[0]
            if current_entry_idx == entry_idx:
                if current_exit_idx in entry_easy[1]:
                    found = True
                    iflower_medium_roundabout_all = True
                    lane = entry_easy[2]
                    break
    if not found:
        for entry_easy in (spawn_points_2_lane_roundabout_small_easy+spawn_points_2_lane_roundabout_small_difficult):
            entry_idx = entry_easy[0]
            if current_entry_idx == entry_idx:
                if current_exit_idx in entry_easy[1]:
                    entryIndx50mRoundabout = [73,15,55,96,58,119,22,16,96,58]
                    found = True
                    lane = entry_easy[2]
                    if current_entry_idx in entryIndx50mRoundabout:
                        if50Roundabout_roundabout_all = True
                    else:
                        if16Roundabout_roundabout_all = True
                    break


    if iflower_medium_roundabout_all:
        return '32m',lane
    elif ifupper_medium_roundabout:
        return '40m',lane
    elif ifroundabout20m:
        return '20m',lane
    elif if16Roundabout_roundabout_all:
        return '16m',lane
    elif if50Roundabout_roundabout_all:
        return '50m',lane
    else:
        raise Exception('fuc')
