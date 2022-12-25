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
    print('HERE')
    print(f"DEG - {angle_deg}")
    print(f"RAD - {angle_rad}")
    #
    # print(f"In calculate_angle_with_center_of_lane")
    # print(f"Previous position X:{previous_position.x} Y:{previous_position.y}")
    # print(f"Current position X:{current_position.x} Y:{current_position.y}")
    # print(f"Next position X:{next_position.x} Y:{next_position.y}")
    # print(f"Output angle in degrees {angle_deg}")

    return angle_deg


def run_tests():
    assert round(calculate_angle_with_center_of_lane((50,20),(10,50),(50,80)),2) == 53.13
    assert round(calculate_angle_with_center_of_lane((10,10), (30,50),(100,70)),2) == 29.74
    assert round(calculate_angle_with_center_of_lane((10,10), (-30,0),(100,70)),2) == 160.35
    assert round(calculate_angle_with_center_of_lane((10,10), (30,0),(50,70)),2) == 82.87
    assert round(calculate_angle_with_center_of_lane((10,10), (40,10),(10,70)),2) == 90.00


    # First quadrant

    assert round(calculate_angle_with_center_of_lane((10,10), (50,50),(10,70)),2) == 45.00
    assert round(calculate_angle_with_center_of_lane((10,10), (50,50),(20,70)),2) == 35.54

    # Vertical

    assert round(calculate_angle_with_center_of_lane((10,10), (10,90),(10,70)),2) == 0
    assert round(calculate_angle_with_center_of_lane((10,10), (10,90),(20,70)),2) == 9.46
    # Second quadrant

    assert round(calculate_angle_with_center_of_lane((10,10), (-40,30),(10,70)),2) == 68.20
    assert round(calculate_angle_with_center_of_lane((10,10), (-40,30),(20,70)),2) == 77.66

    # Third quadrant
    assert round(calculate_angle_with_center_of_lane((10,10), (-20,-50),(10,70)),2) == 153.43
    assert round(calculate_angle_with_center_of_lane((10,10), (-20,-50),(20,70)),2) == 162.90

    # fourth quadrant
    assert round(calculate_angle_with_center_of_lane((10,10), (50,-60),(10,70)),2) == 150.26
    assert round(calculate_angle_with_center_of_lane((10,10), (50,-60),(20,70)),2) == 140.79





