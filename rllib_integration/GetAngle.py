import math
import warnings


def calculate_angle_with_center_of_lane(previous_position, current_position, next_position):
    Vl_x = current_position.x - previous_position.x
    Vl_y = current_position.y - previous_position.y

    Vj_x = next_position.x - previous_position.x
    Vj_y = next_position.y - previous_position.y

    enumerator = (Vl_x * Vj_x) + (Vl_y * Vj_y)
    denominator = math.sqrt(math.pow(Vl_x, 2) + math.pow(Vl_y, 2)) * math.sqrt(math.pow(Vj_x, 2) + math.pow(Vj_y, 2))

    temp = enumerator / denominator
    if temp > 1 or temp < -1:
        temp = np.clip(temp,-1,1)
        # warnings.warn(f'WARNING value of {temp} clipped to {np.clip(temp,-1,1)} ')
    angle_rad = math.acos(temp)
    # angle_deg = angle_rad* 180/math.pi

    #
    # print(f"In calculate_angle_with_center_of_lane")
    # print(f"Previous position X:{previous_position.x} Y:{previous_position.y}")
    # print(f"Current position X:{current_position.x} Y:{current_position.y}")
    # print(f"Next position X:{next_position.x} Y:{next_position.y}")
    # print(f"Output angle in degrees {angle_deg}")

    # Finding on which side the angle
    x = np.array([
        [1, previous_position.x, previous_position.y],
        [1, next_position.x, next_position.y],
        [1, current_position.x, current_position.y]])
    det = np.linalg.det(x)
    if det < 0:
        return -angle_rad
    else:
        return angle_rad


import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(waypoint_forward_vector, vehicle_forward_vector):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            # >>> angle_between((1, 0, 0), (0, 1, 0))
            # 1.5707963267948966
            # >>> angle_between((1, 0, 0), (1, 0, 0))
            # 0.0
            # >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """

    v1_u = unit_vector((waypoint_forward_vector.x, waypoint_forward_vector.y, waypoint_forward_vector.z))
    v2_u = unit_vector((vehicle_forward_vector.x, vehicle_forward_vector.y, vehicle_forward_vector.z))
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    # angle_deg = angle_rad * 180 / math.pi

    # Finding on which side the angle
    point_on_waypoint_forward_vector_1 = Vector(0,0)
    point_on_waypoint_forward_vector_2 = Vector(waypoint_forward_vector.x,waypoint_forward_vector.y)
    point_on_truck_forward_vector_1 = Vector(0.5*vehicle_forward_vector.x,0.5*vehicle_forward_vector.y)

    x = np.array([
        [1,point_on_waypoint_forward_vector_1.x,point_on_waypoint_forward_vector_1.y],
        [1,point_on_waypoint_forward_vector_2.x,point_on_waypoint_forward_vector_2.y],
        [1,point_on_truck_forward_vector_1.x,point_on_truck_forward_vector_1.y]])
    det = np.linalg.det(x)
    if det < 0:
        return -angle_rad
    else:
        return angle_rad


class Vector:
    def __init__(self,x,y,z=0):
        self.x =x
        self.y = y
        self.z = z
def run_tests():
    assert round(calculate_angle_with_center_of_lane(Vector(50, 20), Vector(10, 50), Vector(50, 80)), 2) == 0.93
    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(30, 50), Vector(100, 70)), 2) == 0.52
    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(-30, 0), Vector(100, 70)), 2) == 2.80
    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(30, 0), Vector(50, 70)), 2) == -1.45
    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(40, 10), Vector(10, 70)), 2) == -1.57

    assert round(angle_between(Vector(0, 60),Vector(-40, 30)),2) == 0.93
    assert round(angle_between(Vector(90, 60),Vector(20,40)),2) == 0.52
    assert round(angle_between(Vector(90,60),Vector(-40,-10)),2) == 2.80
    assert round(angle_between(Vector(40,60),Vector(20,-10)),2) == -1.45
    assert round(angle_between(Vector(0,60),Vector(30,0)),2) == -1.57


    # First quadrant

    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(50, 50), Vector(10, 70)), 2) == -0.79
    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(50, 50), Vector(20, 70)), 2) == -0.62

    assert round(angle_between(Vector(0,60),Vector(40,40)),2) == -0.79
    assert round(angle_between(Vector(10,60),Vector(40,40)),2) == -0.62

    # Vertical

    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(10, 90), Vector(10, 70)), 2) == 0
    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(10, 90), Vector(20, 70)), 2) == 0.17
    assert round(angle_between(Vector(0,60),Vector(0,80)),2) == 0
    assert round(angle_between(Vector(10,60),Vector(0,80)),2) == 0.17


    # Second quadrant

    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(-40, 30), Vector(10, 70)), 2) == 1.19
    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(-40, 30), Vector(20, 70)), 2) == 1.36
    assert round(angle_between(Vector(0,60),Vector(-50,20)),2) == 1.19
    assert round(angle_between(Vector(10,60),Vector(-50,20)),2) == 1.36

    # Third quadrant
    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(-20, -50), Vector(10, 70)), 2) == 2.68
    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(-20, -50), Vector(20, 70)), 2) == 2.84
    assert round(angle_between(Vector(0,60),Vector(-30,-60)),2) ==2.68
    assert round(angle_between(Vector(10,60),Vector(-30,-60)),2) ==2.84

    # fourth quadrant
    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(50, -60), Vector(10, 70)), 2) == -2.62
    assert round(calculate_angle_with_center_of_lane(Vector(10, 10), Vector(50, -60), Vector(20, 70)), 2) == -2.46
    assert round(angle_between(Vector(0,60),Vector(40,-70)),2) == -2.62
    assert round(angle_between(Vector(10,60),Vector(40,-70)),2) == -2.46
    print('All assertions passed')

    # assert angle_between
# run_tests()



