import math

def calculate_angle_with_center_of_lane(previous_position, current_position, next_position):
    Vl_x = current_position.x - previous_position.x
    Vl_y = current_position.y - previous_position.y

    Vj_x = next_position.x - previous_position.x
    Vj_y = next_position.y - previous_position.y

    enumerator = (Vl_x * Vj_x) + (Vl_y * Vj_y)
    denominator = math.sqrt(math.pow(Vl_x, 2) + math.pow(Vl_y, 2)) * math.sqrt(math.pow(Vj_x, 2) + math.pow(Vj_y, 2))

    angle_rad = math.acos(enumerator / denominator)
    angle_deg = angle_rad* 180/math.pi

    #
    # print(f"In calculate_angle_with_center_of_lane")
    # print(f"Previous position X:{previous_position.x} Y:{previous_position.y}")
    # print(f"Current position X:{current_position.x} Y:{current_position.y}")
    # print(f"Next position X:{next_position.x} Y:{next_position.y}")
    # print(f"Output angle in degrees {angle_deg}")

    return angle_deg


import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(angle_1, angle_2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            # >>> angle_between((1, 0, 0), (0, 1, 0))
            # 1.5707963267948966
            # >>> angle_between((1, 0, 0), (1, 0, 0))
            # 0.0
            # >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """

    v1_u = unit_vector((angle_1.x,angle_1.y,angle_1.z))
    v2_u = unit_vector((angle_2.x,angle_2.y,angle_2.z))
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_deg = angle_rad * 180 / math.pi
    return angle_deg

class Angle:
    def __init__(self,x,y):
        self.x =x
        self.y = y
def run_tests():
    assert round(calculate_angle_with_center_of_lane(Angle(50,20),Angle(10,50),Angle(50,80)),2) == 53.13
    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(30,50),Angle(100,70)),2) == 29.74
    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(-30,0),Angle(100,70)),2) == 160.35
    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(30,0),Angle(50,70)),2) == 82.87
    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(40,10),Angle(10,70)),2) == 90.00


    # First quadrant

    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(50,50),Angle(10,70)),2) == 45.00
    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(50,50),Angle(20,70)),2) == 35.54

    # Vertical

    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(10,90),Angle(10,70)),2) == 0
    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(10,90),Angle(20,70)),2) == 9.46
    # Second quadrant

    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(-40,30),Angle(10,70)),2) == 68.20
    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(-40,30),Angle(20,70)),2) == 77.66

    # Third quadrant
    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(-20,-50),Angle(10,70)),2) == 153.43
    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(-20,-50),Angle(20,70)),2) == 162.90

    # fourth quadrant
    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(50,-60),Angle(10,70)),2) == 150.26
    assert round(calculate_angle_with_center_of_lane(Angle(10,10), Angle(50,-60),Angle(20,70)),2) == 140.79
    print('All assertions passed')
# run_tests()



