import math
import numpy as np

from rllib_integration.GetAngle import calculate_angle_with_center_of_lane
from carla import Location

class Vector():
    def __init__(self,x_coof, c_coof,y_coof, x_0, y_0, x_1,y_1):
        self.x_coof = x_coof
        self.c_coof = c_coof
        self.y_coof = y_coof
        self.x_0 = x_0
        self.y_0 = y_0
        self.x_1 = x_1
        self.y_1 = y_1
def get_radii(route, last_waypoint_index,no_of_points_to_calculate_chord):
    max_radius_value = 0
    max_radius_value_before_sign = 1
    length_of_output_array = 10
    perpendicular_bisectors = []
    radii = []

    calculate_previous_radii = False
    if calculate_previous_radii:
        # Lets say we have the next 10 points of the route and we want to find its radius

        # Calculating the number of chords we can calculate given in the previous waypoints exist
        no_of_whole_chords_before_current_point = last_waypoint_index//no_of_points_to_calculate_chord
        # Starting 3 radii before current one.
        max_chords_before_current_points = 3
        # If we are ahead that there are plenty of waypoint behind us to calculate three previous chords
        if no_of_whole_chords_before_current_point >= max_chords_before_current_points:
            remaining_waypoints_on_route = route[last_waypoint_index-(no_of_points_to_calculate_chord*max_chords_before_current_points):]
        else:
            # If there isnt enough waypoints, we add the max value for the number of chords we cannot calculate and calculate as much chords as we can
            for i in range(max_chords_before_current_points-no_of_whole_chords_before_current_point):
                radii.append(max_radius_value)
            remaining_waypoints_on_route = route[last_waypoint_index-(no_of_points_to_calculate_chord*no_of_whole_chords_before_current_point):]

    else:
        remaining_waypoints_on_route = route[last_waypoint_index:]
    for i in range(0,len(remaining_waypoints_on_route)-no_of_points_to_calculate_chord,no_of_points_to_calculate_chord):
        x_0 = remaining_waypoints_on_route[i].location.x
        y_0 = remaining_waypoints_on_route[i].location.y

        x_1 = remaining_waypoints_on_route[i+no_of_points_to_calculate_chord].location.x
        y_1 = remaining_waypoints_on_route[i+no_of_points_to_calculate_chord].location.y


        x_p = (x_0 + x_1) /2
        y_p = (y_0 + y_1) /2
        y_coof = 1

        if x_1-x_0 != 0:
            # Not straight line
            m_p = (y_1-y_0) / (x_1-x_0)
            m_p = -1/m_p

        else:
            # Horizontal lines
            print("Horizontal Line")
            m_p = 0

        x_coof = m_p
        c = -x_p * m_p + y_p

        if y_1-y_0 == 0:
            # Vertical line
            print("Vertical Line")
            x_coof = 0
            y_coof = 0
            c = x_p


        perpendicular_bisectors.append(Vector(x_coof=x_coof, c_coof=c,y_coof=y_coof,x_0=x_0,y_0=y_0,x_1=x_1,y_1=y_1))


    for i in range(0,len(perpendicular_bisectors)-1):
        # For vertical lines
        if perpendicular_bisectors[i].y_coof == 0 and perpendicular_bisectors[i+1].y_coof == 0:
            # two vertical lines
            radii.append(max_radius_value)
            continue
        elif perpendicular_bisectors[i].y_coof == 0:
            x_center = perpendicular_bisectors[i].c_coof
            y_center = perpendicular_bisectors[i+1].x_coof*x_center + perpendicular_bisectors[i+1].c_coof
        elif perpendicular_bisectors[i+1].y_coof == 0:
            x_center = perpendicular_bisectors[i+1].c_coof
            y_center = perpendicular_bisectors[i].x_coof*x_center + perpendicular_bisectors[i].c_coof

        # For horizontal lines
        elif perpendicular_bisectors[i].y_coof != 0 and perpendicular_bisectors[i].x_coof == 0 and perpendicular_bisectors[i+1].y_coof != 0 and perpendicular_bisectors[i+1].x_coof == 0:
        #     two horizonatal lines
            radii.append(max_radius_value)
            continue

        elif perpendicular_bisectors[i].y_coof != 0 and perpendicular_bisectors[i].x_coof != 0 and perpendicular_bisectors[i+1].y_coof != 0 and perpendicular_bisectors[i+1].x_coof == 0:
            y_center = perpendicular_bisectors[i+1].c_coof
            x_center = (y_center - perpendicular_bisectors[i].c_coof)/ perpendicular_bisectors[i].x_coof


        elif perpendicular_bisectors[i].y_coof != 0 and perpendicular_bisectors[i].x_coof == 0 and perpendicular_bisectors[i+1].y_coof != 0 and perpendicular_bisectors[i+1].x_coof != 0:
            y_center = perpendicular_bisectors[i].c_coof
            x_center = (y_center - perpendicular_bisectors[i+1].c_coof)/ perpendicular_bisectors[i+1].x_coof

        else:
            # Find where two perpendicular bisectiors meet
            if (perpendicular_bisectors[i].x_coof-perpendicular_bisectors[i+1].x_coof) == 0:
                radii.append(max_radius_value)
                continue
            else:
                x_center = (perpendicular_bisectors[i+1].c_coof - perpendicular_bisectors[i].c_coof) / (perpendicular_bisectors[i].x_coof-perpendicular_bisectors[i+1].x_coof)
                y_center = perpendicular_bisectors[i].x_coof*x_center + perpendicular_bisectors[i].c_coof


        r_0 = math.sqrt((x_center-perpendicular_bisectors[i].x_0)**2+(y_center-perpendicular_bisectors[i].y_0)**2)/100
        r_1 = math.sqrt((x_center-perpendicular_bisectors[i+1].x_1)**2+(y_center-perpendicular_bisectors[i+1].y_1)**2)/100

        angle_between_starting_ending_point = calculate_angle_with_center_of_lane(
            previous_position=Location(x=perpendicular_bisectors[i].x_0,y=perpendicular_bisectors[i].y_0,z=0),
            current_position=Location(x=perpendicular_bisectors[i].x_1,y=perpendicular_bisectors[i].y_1,z=0),
            next_position=Location(x=perpendicular_bisectors[i+1].x_1,y=perpendicular_bisectors[i+1].y_1,z=0))

        sign = 1
        if angle_between_starting_ending_point < 0:
            sign = -1

        assert max_radius_value_before_sign == 1
        riadus = np.clip((r_0 + r_1) / 2, 0, max_radius_value_before_sign)

        # If higher value mean tighter radii, the max value needs to be 0
        signed_radius = False
        if signed_radius:
            assert max_radius_value == 0
            riadus = 1-riadus

            radii.append(sign * riadus)
        else:
            radii.append(riadus)

        if len(radii) > length_of_output_array:
            break

    if len(radii) < length_of_output_array:
        for i in range(length_of_output_array-len(radii)):
            radii.append(max_radius_value)

    return radii[0:length_of_output_array]