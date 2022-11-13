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
for x in range(100):
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)
    # plt.(range(0,10))
    # plt.yticks(range(0,10))
    plt.xlim([-12,12])
    plt.ylim([-12,12])

    x_pos = random.uniform(-10,10)
    y_pos = random.uniform(-10,10)

    x_last = random.uniform(-10,10)
    y_last = random.uniform(-10,10)

    x_next = random.uniform(-10,10)
    y_next = random.uniform(-10,10)

    print(f"x_pos {x_pos} y_pos {y_pos}")
    print(f"x_last {x_last} y_last {y_last}")
    print(f"x_next {x_next} y_next {y_next}")


    plt.plot([x_last,x_next],[y_last,y_next] )
    # plotting the points
    plt.plot(x_pos, y_pos, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green",)


    val = update_next_waypoint(x_pos,y_pos,x_last,y_last,x_next,y_next)

    if x_next-x_last == 0:
        x= []
        y = []
        for a in range(-20,20):
            x.append(a)
            y.append(y_last)
        plt.plot(x, y)
    elif y_next-y_last == 0:
        x = []
        y= []
        for a in range(-20,20):
            x.append(x_last)
            y.append(a)
        plt.plot(x,y)

    else:
        gradOfPrevToNext = (y_next-y_last) / (x_next-x_last)
        gradOfPerpendicular = -1/gradOfPrevToNext
        cOfPerpendicular = y_last - gradOfPerpendicular*x_last
        print(f"gradOfPerpendicular {gradOfPerpendicular}")
        print(f"cOfPerpendicular {cOfPerpendicular}")
        x = []
        y= []
        for a in range(-20,20):
            x.append(a)
            y.append(gradOfPerpendicular*a+cOfPerpendicular)
        plt.plot(x,y,label="Perpendicular")
    leg = plt.legend(loc='upper center')
    if val == 0:
        print('POINT ON LINE')
        plt.title(f"Result = ONLINE")
    if val == 1:
        plt.title(f"Result = FORWARD")
    if val == -1:
        plt.title(f"Result = BACKWARD")
    print('--------------------')

    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')


    # giving a title to my graph

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