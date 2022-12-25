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
def plot_points(previous_position, current_position, next_position, current_waypoint, next_waypoint,in_front_of_waypoint):


    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)
    # plt.(range(0,10))
    # plt.yticks(range(0,10))
    plt.xlim([0.45,0.6])
    plt.ylim([0.6,0.7])

    print(f"x_pos {current_position.x} y_pos {current_position.y}")
    print(f"x_last {previous_position.x} y_last {previous_position.y}")
    print(f"x_next {next_position.x} y_next {next_position.y}")


    plt.plot([previous_position.x,next_position.x],[previous_position.y,next_position.y] )
    # plotting the points
    plt.plot(current_position.x, current_position.y, marker="o", markersize=3, markeredgecolor="red", markerfacecolor="red",)
    plt.plot(current_waypoint.x, current_waypoint.y, marker="o", markersize=3, markeredgecolor="black", markerfacecolor="black",)
    plt.plot(next_waypoint.x, next_waypoint.y, marker="o", markersize=3, markeredgecolor="blue", markerfacecolor="blue",)


    # val = update_next_waypoint(current_position.x,current_position.y,previous_position.x,previous_position.y,next_position.x,next_position.y)

    if next_position.x-previous_position.x == 0:
        x= []
        y = []
        for a in range(-20,20):
            x.append(a)
            y.append(previous_position.y)
        plt.plot(x, y)
    elif next_position.y-previous_position.y == 0:
        x = []
        y= []
        for a in range(-20,20):
            x.append(previous_position.x)
            y.append(a)
        plt.plot(x,y)

    else:
        gradOfPrevToNext = (next_position.y-previous_position.y) / (next_position.x-previous_position.x)
        gradOfPerpendicular = -1/gradOfPrevToNext
        cOfPerpendicular = previous_position.y - gradOfPerpendicular*previous_position.x
        print(f"gradOfPerpendicular {gradOfPerpendicular}")
        print(f"cOfPerpendicular {cOfPerpendicular}")
        x = []
        y= []
        for a in range(-20,20):
            x.append(a)
            y.append(gradOfPerpendicular*a+cOfPerpendicular)
        plt.plot(x,y,label="Perpendicular")
    leg = plt.legend(loc='upper center')
    if in_front_of_waypoint == 0:
        print('POINT ON LINE')
        plt.title(f"Result = ONLINE")
    if in_front_of_waypoint == 1:
        plt.title(f"Result = FORWARD")
    if in_front_of_waypoint == -1:
        plt.title(f"Result = BACKWARD")
    print('--------------------')

    # naming the x-axis
    plt.xlabel('x - axis')
    # naming the y-axis
    plt.ylabel('y - axis')

    # function to show the plot
    plt.show()



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