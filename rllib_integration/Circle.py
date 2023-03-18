import math
import numpy as np
class Vector():
    def __init__(self,x_coof, c_coof,y_coof, x_0, y_0, x_1,y_1):
        self.x_coof = x_coof
        self.c_coof = c_coof
        self.y_coof = y_coof
        self.x_0 = x_0
        self.y_0 = y_0
        self.x_1 = x_1
        self.y_1 = y_1
def get_radii(remaining_waypoints_on_route,no_of_points_to_calculate_chord):
    # Lets say we have the next 10 points of the route and we want to find its radius
    perpendicular_bisectors = []
    radii = []
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

    max_radius_value = 1
    length_of_output_array = 6
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


        r_0 = math.sqrt((x_center-perpendicular_bisectors[i].x_0)**2+(y_center-perpendicular_bisectors[i].y_0)**2)
        r_1 = math.sqrt((x_center-perpendicular_bisectors[i+1].x_1)**2+(y_center-perpendicular_bisectors[i+1].y_1)**2)

        radii.append(np.clip((r_0+r_1)/2,0,max_radius_value)/100)

    if len(radii) < length_of_output_array:
        for i in range(length_of_output_array-len(radii)):
            radii.append(max_radius_value)

    return radii[0:6]