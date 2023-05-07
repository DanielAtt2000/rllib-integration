# COPY OVER LATEST METHOD
import random

def update_next_waypoint( x_pos, y_pos, last_x, last_y, next_x, next_y):
    last_to_next_vector = (next_x - last_x, next_y - last_y)

    if last_x == next_x and last_y == next_y:
        raise Exception('This should never be the case')

    if last_to_next_vector[0] == 0:
        # Vertical line between waypoints
        if y_pos == last_y:
            return 0
        elif y_pos < last_y:
            if next_y < last_y:
                return 1
            elif next_y > last_y:
                return -1
        elif y_pos > last_y:
            if next_y < last_y:
                return -1
            elif next_y > last_y:
                return 1

    elif last_to_next_vector[1] == 0:
        # Horizontal line between waypoints
        if x_pos == last_x:
            return 0
        elif x_pos < last_x:
            if next_x < last_x:
                return 1
            elif next_x > last_x:
                return -1
        elif x_pos > last_x:
            if next_x < last_x:
                return -1
            elif next_x > last_x:
                return 1

    a = 1
    t = 2
    b = (-last_to_next_vector[0] / last_to_next_vector[1]) * a

    # Equation of perpendicular line
    # r = ( last_x, last_y) +  t * (a,b)
    x_on_perpendicular = last_x + t * a
    y_on_perpendicular = last_y + t * b

    d_pos = (x_pos - last_x) * (y_on_perpendicular - last_y) - (y_pos - last_y) * (x_on_perpendicular - last_x)
    d_infront = (next_x - last_x) * (y_on_perpendicular - last_y) - (next_y - last_y) * (
            x_on_perpendicular - last_x)

    if d_pos == 0:
        # Vehicle is on the line
        return 0
    elif (d_pos > 0 and d_infront > 0) or (d_pos < 0 and d_infront < 0):
        # Vehicle skipped line
        return 1
    else:
        return -1


assert update_next_waypoint(7,4,6,1,1,0) == -1

# assert update_next_waypoint2(0,5,10,3,10,1) == -1

# assert update_next_waypoint(5,10,-5,5,10,20) == 1
# assert update_next_waypoint(0,0,-5,5,10,20) == 0
# assert update_next_waypoint(-5,0,-5,5,10,20) == -1
#
# assert update_next_waypoint(5,10,10,5,10,20) == 1
# assert update_next_waypoint(0,0,10,5,10,20) == -1
# assert update_next_waypoint(-5,0,10,5,10,20) == -1


# importing the required module
import matplotlib.pyplot as plt
def plot_points(previous_position, current_position, next_position, current_waypoint, next_waypoint,in_front_of_waypoint=-5,angle=-1):


    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)
    # plt.(range(0,10))
    # plt.yticks(range(0,10))
    x_values = [previous_position.x,current_position.x,next_position.x,current_waypoint.x,next_waypoint.x]
    y_values = [previous_position.y,current_position.y,next_position.y,current_waypoint.y,next_waypoint.y]
    x_min = min(x_values)
    x_max = max(x_values)

    y_min = min(y_values)
    y_max = max(y_values)
    buffer = 0.5

    plt.xlim([x_min - buffer,x_max + buffer])
    plt.ylim([y_min - buffer,y_max + buffer])

    # print(f"x_pos {current_position.x} y_pos {current_position.y}")
    # print(f"x_last {previous_position.x} y_last {previous_position.y}")
    # print(f"x_next {next_position.x} y_next {next_position.y}")


    plt.plot([previous_position.x,next_position.x],[previous_position.y,next_position.y] )
    # plotting the points
    plt.plot(current_position.x, current_position.y, marker="^", markersize=3, markeredgecolor="red", markerfacecolor="red",label='Current Vehicle Position')
    plt.plot(current_waypoint.x, current_waypoint.y, marker="o", markersize=3, markeredgecolor="black", markerfacecolor="black",label='Current Waypoint')
    plt.plot(next_waypoint.x, next_waypoint.y, marker="o", markersize=3, markeredgecolor="blue", markerfacecolor="blue",label='Next Waypoint')
    plt.plot(previous_position.x, previous_position.y, marker="o", markersize=3, markeredgecolor="green", markerfacecolor="green",label='Previous Waypoint')
    plt.plot(next_position.x, next_position.y, marker="o", markersize=3, markeredgecolor="yellow", markerfacecolor="yellow",label='Next Position')
    plt.plot([current_position.x,previous_position.x],[current_position.y,previous_position.y])
    plt.plot([current_position.x,next_position.x],[current_position.y,next_position.y])
    plt.gca().invert_yaxis()
    # val = update_next_waypoint(current_position.x,current_position.y,previous_position.x,previous_position.y,next_position.x,next_position.y)



    if next_waypoint.x-current_waypoint.x == 0:
        x= []
        y = []
        for a in range(-20,20):
            x.append(a)
            y.append(current_waypoint.y)
        plt.plot(x, y)
    elif next_waypoint.y-current_waypoint.y == 0:
        x = []
        y= []
        for a in range(-20,20):
            x.append(current_waypoint.x)
            y.append(a)
        plt.plot(x,y)

    else:
        gradOfPrevToNext = (next_waypoint.y-current_waypoint.y) / (next_waypoint.x-current_waypoint.x)
        gradOfPerpendicular = -1/gradOfPrevToNext
        cOfPerpendicular = current_waypoint.y - gradOfPerpendicular*current_waypoint.x
        print(f"gradOfPerpendicular {gradOfPerpendicular}")
        print(f"cOfPerpendicular {cOfPerpendicular}")
        x = []
        y= []
        for a in range(-20,20):
            x.append(a)
            y.append(gradOfPerpendicular*a+cOfPerpendicular)
        plt.plot(x,y,label="Perpendicular")
    leg = plt.legend(loc='upper right')
    if in_front_of_waypoint == 0:
        print('POINT ON LINE')
        plt.title(f"Result = ONLINE - {angle}")
    if in_front_of_waypoint == 1:
        plt.title(f"Result = FORWARD - {angle}")
    if in_front_of_waypoint == -1:
        plt.title(f"Result = BACKWARD - {angle}")
    print('--------------------')

    # naming the x-axis
    plt.xlabel('x - axis')
    # naming the y-axis
    plt.ylabel('y - axis')

    # function to show the plot
    plt.show()


def plot_route(route,last_waypoint_index=-1,truck_transform=-1,number_of_waypoints_ahead_to_calculate_with=-1):
    x_route = []
    y_route = []
    for point in route:
        # print(f"X: {point.location.x} Y:{point.location.y}")
        x_route.append(point.location.x)
        y_route.append(point.location.y)

    x_min = min(x_route)
    x_max = max(x_route)

    y_min = min(y_route)
    y_max = max(y_route)
    buffer = 10

    # print(f"X_TRUCK: {truck_normalised_transform.location.x} Y_TRUCK {truck_normalised_transform.location.y}")
    plt.plot([x_route.pop(0)], y_route.pop(0), 'bo')
    plt.plot(x_route, y_route, 'y^')
    plt.plot([route[0].location.x], [route[0].location.y], 'ro', label='Starting waypoint')
    if last_waypoint_index != -1:
        plt.plot([route[last_waypoint_index - 1].location.x],
                 [route[last_waypoint_index - 1].location.y], 'ro', label='Previous Waypoint')
    if truck_transform != -1:
        plt.plot([truck_transform.location.x], [truck_transform.location.y], 'gs', label='Current Vehicle Location')
    if last_waypoint_index != -1 and number_of_waypoints_ahead_to_calculate_with != -1:
        plt.plot([route[last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location.x],
                 [route[last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location.y], 'bo',
                 label=f"{number_of_waypoints_ahead_to_calculate_with} waypoints ahead")
    plt.axis([0, 250, 0 , 250 ])
    # plt.axis([0, 1, 0, 1])
    # plt.title(f'{angle_to_center_of_lane_degrees * 180}')
    plt.gca().invert_yaxis()
    plt.legend(loc='upper center')
    plt.show()

def draw_route_in_order(route):
    x_route = []
    y_route = []
    x_min = 10000000
    x_max = -1000000
    y_min = 10000000
    y_max = -1000000
    
    for point in route:
        # print(f"X: {point.location.x} Y:{point.location.y}")
        x_min = min(x_min,point.location.x)
        x_max = max(x_max,point.location.x)
        y_min = min(y_min,point.location.y)
        y_max = max(y_max,point.location.y)
        x_route.append(point.location.x)
        y_route.append(point.location.y)


    last_point_plotted = 0

    for p,point in enumerate(route):
        plt.plot(x_route, y_route, 'g^')
        for i in range(last_point_plotted):
            plt.plot(x_route[i],y_route[i],'ro')
        plt.plot(x_route[last_point_plotted], y_route[last_point_plotted], 'yo')
        last_point_plotted +=1

        buffer = 10
        # plt.axis([145,165, 100,130])
        plt.axis([x_min-buffer, x_max+buffer, y_min-buffer, y_max+buffer])
        # plt.axis([0, 1, 0, 1])
        # plt.title(f'{angle_to_center_of_lane_degrees * 180}')
        plt.gca().invert_yaxis()
        # plt.legend(loc='upper center')
        plt.show()


def plot_all_routes(all_routes=-1):
    x_routes = []
    y_routes = []
    for route in all_routes:
        temp_x_route = []
        temp_y_route = []
        for point in route:
            # print(f"X: {point.location.x} Y:{point.location.y}")
            temp_x_route.append(point.location.x)
            temp_y_route.append(point.location.y)
        x_routes.append(temp_x_route)
        y_routes.append(temp_y_route)

    def minimum(temp_list):
        min = 1000000
        for item in temp_list:
            for element in item:
                if element < min:
                    min = element

        return min

    def maxiumum(temp_list):
        max = -1000000
        for item in temp_list:
            for element in item:
                if element > max:
                    max = element

        return max
    x_min = minimum(x_routes)
    x_max = maxiumum(x_routes)

    y_min = minimum(y_routes)
    y_max = maxiumum(y_routes)
    buffer = 10

    # print(f"X_TRUCK: {truck_normalised_transform.location.x} Y_TRUCK {truck_normalised_transform.location.y}")
    idx = 0
    idx_2 = 0
    for x_route, y_route in zip(x_routes,y_routes):
        idx_2 = 0
        for x_route_2,y_route_2 in zip(x_routes,y_routes):
            if idx_2 != idx:
                plt.plot(x_route_2, y_route_2, 'go')
            idx_2 +=1

        plt.plot([x_route.pop(0)], y_route.pop(0), 'bo')
        plt.plot(x_route, y_route, 'y^')
        plt.plot([x_route[0]], [y_route[0]], 'ro', label='Starting waypoint')
        plt.plot([x_route[len(x_route)-15]], [y_route[len(y_route)-15]], 'r^', label='Ending waypoint')
        plt.axis([x_min - buffer, x_max + buffer, y_min - buffer, y_max + buffer])
        # plt.axis([0, 1, 0, 1])
        # plt.title(f'{angle_to_center_of_lane_degrees * 180}')
        plt.gca().invert_yaxis()
        plt.legend(loc='upper center')
        plt.show()

        idx +=1

# TO CHEKC

# x_pos 5 y_pos 5
# x_last 5 y_last 5
# x_next 1 y_next 10

# x_pos 6 y_pos 2
# x_last 4 y_last 3
# x_next 10 y_next 3
# --------------------


# x_pos 2 y_pos 3
# x_last 7 y_last 8
# x_next 7 y_next 1

# x_pos 1 y_pos 3
# x_last 5 y_last 5
# x_next 10 y_next 9
# gradOfPerpendicular -1.25
# cOfPerpendicular 11.25

# x_pos 7 y_pos 4
# x_last 6 y_last 1
# x_next 1 y_next 0
# gradOfPerpendicular -5.0
# cOfPerpendicular 31.0
# --------------------