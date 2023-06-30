#!/usr/bin/env python
import math
# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import subprocess
import os
import random
import signal
import time
import pickle
import numpy as np
import psutil
import logging
import socket
import datetime
import carla
from matplotlib import pyplot as plt

import pyautogui

from rllib_integration.LineIntersection import Point, lineLineIntersection
from rllib_integration.RouteGeneration.global_route_planner import GlobalRoutePlanner
from Helper import open_pickle, save_to_pickle
from rllib_integration.sensors.sensor_interface import SensorInterface
from rllib_integration.sensors.factory import SensorFactory
from rllib_integration.helper import join_dicts
from rllib_integration.GetStartStopLocation import get_entry_exit_spawn_point_indices, \
    get_entry_exit_spawn_point_indices_2_lane, visualise_all_routes

from rllib_integration.TestingWayPointUpdater import plot_route

BASE_CORE_CONFIG = {
    # "host": 'localhost',  # Client host
    # "timeout": 10.0,  # Timeout of the client
    # "timestep": 0.05,  # Time step of the simulation
    # "retries_on_error": 10,  # Number of tries to connect to the client
    # "resolution_x": 600,  # Width of the server spectator camera
    # "resolution_y": 600,  # Height of the server spectator camera
    # "quality_level": "Low",  # Quality level of the simulation. Can be 'Low', 'High', 'Epic'
    # "enable_map_assets": False,  # enable / disable all town assets except for the road
    # "enable_rendering": False,  # enable / disable camera images
    # "show_display": False  # Whether or not the server will be displayed
}


def is_used(port):
    """Checks whether or not a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]

def kill_all_servers():
    """Kill all PIDs that start with Carla"""
    processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class CarlaCore:
    """
    Class responsible of handling all the different CARLA functionalities, such as server-client connecting,
    actor spawning and getting the sensors data.
    """
    def __init__(self, config={}):
        """Initialize the server and client"""
        self.client = None
        self.world = None
        self.map = None
        self.hero = None
        self.hero_trailer = None
        self.config = join_dicts(BASE_CORE_CONFIG, config)
        self.sensor_interface_truck = SensorInterface()
        self.sensor_interface_trailer = SensorInterface()
        self.server_port = 2000
        self.server_port_lines = ''
        self.visualise_all_routes = False
        self.times_crazy = []
        self.custom_enable_rendering = False
        self.one_after_the_other = False
        self.total_number_of_routes = -1

        self.route = []
        self.route_points = []
        self.last_waypoint_index = None

        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.last_chosen_route = -2

        self.current_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        # os.mkdir(os.path.join("results", "run_" + str(self.current_time)))

        self.entry_spawn_point_index = -1
        self.exit_spawn_point_index = -1
        self.route_lane = ""
        self.last_roundabout_choice = 1
        self.chosen_routes = {}

        self.in_editor = False

        server_maps = open_pickle('server_maps')
        server_map = server_maps.pop(0)
        save_to_pickle('server_maps',server_maps)

        print(f"Using server map for {server_map}")
        self.map_name = server_map


        if not self.in_editor:
            self.init_server()
        self.connect_client()

    # def init_server(self):
    #     with open('../ppo_example/server_ports.txt','r') as portsFileRead:
    #         self.server_port_lines = portsFileRead.readlines()

    #     self.server_port = int(self.server_port_lines[0])

    #     with open('../ppo_example/server_ports.txt', 'w') as portsFileWrite:
    #         for line in self.server_port_lines:
    #             if str(self.server_port) not in line:
    #                 portsFileWrite.write(line)

    def init_server(self):
        """Start a server on a random port"""
        self.server_port = random.randint(15000, 32000)
        time.sleep(random.randint(0,10))


        # Ray tends to start all processes simultaneously. Use random delays to avoid problems
        # time.sleep(self.server_port/300)

        uses_server_port = is_used(self.server_port)
        uses_stream_port = is_used(self.server_port + 1)
        while uses_server_port and uses_stream_port:
            if uses_server_port:
                print("Is using the server port: " + self.server_port)
            if uses_stream_port:
                print("Is using the streaming port: " + str(self.server_port+1))
            self.server_port += 2
            uses_server_port = is_used(self.server_port)
            uses_stream_port = is_used(self.server_port+1)

        if self.config["show_display"]:
            server_command = [
                "{}/CarlaUE4.sh".format(self.config['carla_location']),
                "-windowed",
                "-ResX={}".format(self.config["resolution_x"]),
                "-ResY={}".format(self.config["resolution_y"]),
            ]
        else:
            server_command = [
                "DISPLAY= ",
                "{}/CarlaUE4.sh".format(self.config['carla_location']),
                "-opengl"  # no-display isn't supported for Unreal 4.24 with vulkan
            ]

        # map_name = "doubleRoundabout37"
        # map_name = "20m"
        # map_name = "mediumRoundabout4"
        server_command += [
            "--carla-rpc-port={}".format(self.server_port),
            "-quality-level={}".format(self.config["quality_level"]),
            "--map={}".format(self.map_name),
            "--no-rendering"
        ]
        print(f'Selected Port {self.server_port}')
        server_command_text = " ".join(map(str, server_command))
        print(server_command_text)
        server_process = subprocess.Popen(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )

        time.sleep(10)

        server_command_2 = [
            "{}/PythonAPI/util/config.py".format(self.config['carla_location']),
            "--port {}".format(self.server_port),
            "--weather {}".format("Default"),
            "--map {}".format(self.map_name),
        ]
        server_command_text_2 = " ".join(map(str, server_command_2))

        server_process = subprocess.Popen(
            server_command_text_2,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )

        time.sleep(5)
        print('Waited 5 seconds for server')
    def save_to_pickle(self,filename, data):
        filename = filename + '.pickle'
        with open(f'{filename}', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def open_pickle(self,filename):
        with open(filename + '.pickle', 'rb') as handle:
            return pickle.load(handle)

    def get_server_port(self):
        possible_ports = [2000,2001]
        try:
            server_ports = self.open_pickle('server_ports.pickle')
            for port in server_ports:
                possible_ports.remove(port)
            if len(possible_ports) != 0:
                return possible_ports[0]
            else:
                raise Exception('We should be here SERVER PORT')
        except:
            self.save_to_pickle('server_ports',[2000])
            return 2000

    def kill_carla(self):
        print("Killing process")
        processes = [p for p in psutil.process_iter() if ("ue4editor" in p.name().lower() or "crashreportclient" in p.name().lower())]
        for process in processes:
            os.kill(process.pid, signal.SIGKILL)

    def open_carla(self,map_name):
        time_to_sleep_between_commands = 3

        time.sleep(time_to_sleep_between_commands)
        unreal_engine_dir = "/home/daniel/carla/Unreal/CarlaUE4/"
        os.chdir(unreal_engine_dir)
        os.system("xdg-open .")
        time.sleep(time_to_sleep_between_commands)
        pyautogui.moveTo(700, 200)
        time.sleep(time_to_sleep_between_commands)
        pyautogui.doubleClick()
        time.sleep(150)

        pyautogui.moveTo(150, 600)
        time.sleep(time_to_sleep_between_commands)
        pyautogui.click()
        time.sleep(time_to_sleep_between_commands)
        time.sleep(5)
        print('Moved Cursor')
        pyautogui.hotkey('ctrl', 'o', interval=0.25)
        time.sleep(time_to_sleep_between_commands)
        pyautogui.write(map_name, interval=0.25)
        time.sleep(time_to_sleep_between_commands)
        pyautogui.press('enter')
        time.sleep(time_to_sleep_between_commands)
        pyautogui.press('esc')
        time.sleep(time_to_sleep_between_commands)
        pyautogui.hotkey('alt', 'p', interval=0.25)
        time.sleep(40)

    def connect_client(self):
        """Connect to the client"""

        for i in range(self.config["retries_on_error"]):
            try:
                file = open('rllib_integration/enable_rendering.txt', 'r')

                if str(file.readline()) == 'False':
                    self.custom_enable_rendering = False
                else:
                    self.custom_enable_rendering = True

                if self.in_editor:
                    self.server_port = 2000

                self.client = carla.Client(self.config["host"], self.server_port)
                self.client.set_timeout(self.config["timeout"])
                self.world = self.client.get_world()

                settings = self.world.get_settings()
                settings.no_rendering_mode = not self.custom_enable_rendering
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.config["timestep"]
                self.world.apply_settings(settings)
                self.world.tick()

                return

            except Exception as e:
                print(" Waiting for server to be ready: {}, attempt {} of {}".format(e, i + 1, self.config["retries_on_error"]))
                time.sleep(3)

        raise Exception("Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration")

        # for i in range(self.config["retries_on_error"]):
        #     try:
        #         # kill_all_servers()
        #
        #         self.client = carla.Client(self.config["host"], self.server_port)
        #
        #     except Exception as e:
        #         print(" FAILED TO CONNECT TO CLIENT: {}, attempt {} of {}".format(e, i + 1, self.config["retries_on_error"]))
        #         time.sleep(3)
        #
        #     try:
        #         print(f"Trying to set up client {i+1} time")
        #         import os
        #         print(os.getcwd())
        #         file = open('rllib_integration/enable_rendering.txt','r')
        #
        #         if str(file.readline()) == 'False':
        #             self.custom_enable_rendering = False
        #         else:
        #             self.custom_enable_rendering = True
        #
        #         self.client.set_timeout(self.config["timeout"])
        #         time.sleep(0.2)
        #         self.world = self.client.get_world()
        #         time.sleep(0.2)
        #         settings = self.world.get_settings()
        #         time.sleep(0.2)
        #         settings.no_rendering_mode = not self.custom_enable_rendering
        #         time.sleep(0.2)
        #         settings.synchronous_mode = True
        #         time.sleep(0.2)
        #         settings.fixed_delta_seconds = self.config["timestep"]
        #         time.sleep(0.2)
        #         self.world.apply_settings(settings)
        #         time.sleep(0.5)
        #         self.world.tick()
        #         time.sleep(0.5)
        #         return
        #
        #     except Exception as e:
        #         print(" FAILED TO STEP UP CLIENT: {}, attempt {} of {}".format(e, i + 1, self.config["retries_on_error"]))
        #         time.sleep(3)
        #
        #     try:
        #
        #         time.sleep(5)
        #
        #         self.kill_carla()
        #         time.sleep(5)
        #         self.open_carla("doubleRoundabout37")
        #         time.sleep(10)
        #
        #         # # Create a socket instance
        #         # socketObject = socket.socket()
        #         #
        #         # # Using the socket connect to a server...in this case localhost
        #         # socketObject.connect((self.config["host"], int(self.config["programPort"])))
        #         # print("Connected to server")
        #         # # Send a message to the web server to supply a page as given by Host param of GET request
        #         # HTTPMessage = "restart"
        #         # bytes = str.encode(HTTPMessage)
        #         # socketObject.sendall(bytes)
        #         #
        #         # # Receive the data
        #         # while (True):
        #         #     data = socketObject.recv(1024)
        #         #     print(data)
        #         #     break
        #         #
        #         # socketObject.close()
        #     except Exception as e:
        #         print(f"Failed to restart carla,{e}")
        #         time.sleep(3)
        #
        # raise Exception("Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration")

    def setup_experiment(self, experiment_config):
        """Initialize the hero and sensors"""

        self.world = self.client.load_world(
            map_name=self.map_name,
            reset_settings=False,
            # map_layers = carla.MapLayer.All if self.config["enable_map_assets"] else carla.MapLayer.NONE
            map_layers=carla.MapLayer.NONE
        )

        self.map = self.world.get_map()

        # Choose the weather of the simulation
        weather = getattr(carla.WeatherParameters, "Default")
        self.world.set_weather(weather)
        #
        # self.tm_port = self.server_port // 10 + self.server_port % 10
        # while is_used(self.tm_port):
        #     print("Traffic manager's port " + str(self.tm_port) + " is already being used. Checking the next one")
        #     self.tm_port += 1
        # print("Traffic manager connected to port " + str(self.tm_port))
        #
        # self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        # self.traffic_manager.set_hybrid_physics_mode(experiment_config["background_activity"]["tm_hybrid_mode"])
        # seed = experiment_config["background_activity"]["seed"]
        # if seed is not None:
        #     self.traffic_manager.set_random_device_seed(seed)
        #
        # # Spawn the background activity
        # self.spawn_npcs(
        #     experiment_config["background_activity"]["n_vehicles"],
        #     experiment_config["background_activity"]["n_walkers"],
        # )

    def set_map_normalisation(self):
        map_buffer = self.config["map_buffer"]
        spawn_points = list(self.map.get_spawn_points())

        min_x = min_y = 1000000
        max_x = max_y = -1000000

        for spawn_point in spawn_points:
            min_x = min(min_x, spawn_point.location.x)
            max_x = max(max_x, spawn_point.location.x)

            min_y = min(min_y, spawn_point.location.y)
            max_y = max(max_y, spawn_point.location.y)

        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2

        x_buffer = (max_x - center_x) * map_buffer
        y_buffer = (max_y - center_y) * map_buffer

        self.min_x = center_x - x_buffer
        self.max_x = center_x + x_buffer

        self.min_y = center_y - y_buffer
        self.max_y = center_y + y_buffer

    def normalise_map_location(self, value, axis,assertBetween=True):
        assert self.min_x != None and self.min_y != None and self.max_x != None and self.max_y != None
        if axis == 'x':
            x_value_normalised = (value - self.min_x) / (self.max_x - self.min_x)
            if (0 <= x_value_normalised <= 1) or not assertBetween:
                return x_value_normalised
            else:
                raise Exception(f"x_value_normalised of out of range between 0 and 1. "
                                f"Input Value: {value}, Normalised Value: {x_value_normalised}")

        elif axis == 'y':
            y_value_normalised = (value - self.min_y) / (self.max_y - self.min_y)
            if (0 <= y_value_normalised <= 1) or not assertBetween:
                return y_value_normalised
            else:
                raise Exception(f"y_value_normalised of out of range between 0 and 1."
                                f" Input Value: {value}, Normalised Value: {y_value_normalised}")

        else:
            raise Exception('Invalid axis given')

    # def get_perpendicular_line_to_point_index(self, point_idx):
    #     from numpy import ones, vstack
    #     from numpy.linalg import lstsq
    #     current_point = self.route[point_idx]
    #     next_point = self.route[point_idx + 1]
    #
    #     points = [(current_point.x, current_point.y), (next_point.x, next_point.y)]
    #     x_coords, y_coords = zip(*points)
    #     A = vstack([x_coords, ones(len(x_coords))]).T
    #     m, c = lstsq(A, y_coords)[0]
    #
    #     m_perpendicular = -1 / m
    #
    #     c_perpendicular = current_point.y - m_perpendicular * current_point.x

    def is_in_front_of_waypoint(self, x_pos, y_pos):
        last_x = self.route[self.last_waypoint_index].location.x
        last_y = self.route[self.last_waypoint_index].location.y

        next_x = self.route[self.last_waypoint_index + 1].location.x
        next_y = self.route[self.last_waypoint_index + 1].location.y

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

    def is_in_front_of_waypoint_from_vector(self,line_point,vector_line, in_front_point):
        # https://stackoverflow.com/questions/22668659/calculate-on-which-side-of-a-line-a-point-is
        # Vector equation of the line is
        #  r = point on line + t(parallel line)

        x_0 = line_point.location.x
        y_0 = line_point.location.y
        t = 2
        x_1 = x_0 + t*vector_line.x
        y_1 = y_0 + t*vector_line.y

        x_p = in_front_point.location.x
        y_p = in_front_point.location.y

        d_pos = (x_1-x_0)*(y_p-y_0) - (x_p-x_0)*(y_1-y_0)

        if d_pos == 0:
            # point on line
            return 0
        elif d_pos > 0:
            # Point behind line
            return 1
        elif d_pos < 0:
            # Point in front of line
            return -1
        else:
            raise Exception("INVALID POSITION")




    def get_perpendicular_distance_between_truck_waypoint_line(self, truck_transform, waypoint_plus_current):
        # https://www.nagwa.com/en/explainers/939127418581/
        # D = magnitude(AP x d) / magnitude(d)
        # d = direction vector of line
        # A is point on line
        # P is point from which to calculate

        def magnitude(vector):
            return math.sqrt(sum(pow(element, 2) for element in vector))

        waypoint_right_vector = self.route[self.last_waypoint_index + waypoint_plus_current].get_right_vector()

        # AP
        ap_x = truck_transform.location.x - self.route[self.last_waypoint_index + waypoint_plus_current].location.x
        ap_y = truck_transform.location.y - self.route[self.last_waypoint_index + waypoint_plus_current].location.y
        ap_z = truck_transform.location.z - self.route[self.last_waypoint_index + waypoint_plus_current].location.z

        AP_vector = [ap_x,ap_y,ap_z]

        d = waypoint_right_vector
        d_vector = [d.x, d.y,d.z]

        ap_cross_d = np.cross(AP_vector,d_vector)

        magnitude_ap_cross_d = magnitude(ap_cross_d)

        distance = magnitude_ap_cross_d / magnitude(d_vector)

        test = False
        if test:
            print(f"waypoint_right_vector {waypoint_right_vector}")
            print(f"waypoint x {self.route[self.last_waypoint_index + waypoint_plus_current].location.x}")
            print(f"waypoint y {self.route[self.last_waypoint_index + waypoint_plus_current].location.y}")
            print(f"waypoint z {self.route[self.last_waypoint_index + waypoint_plus_current].location.z}")
            print(f"truck x {truck_transform.location.x}")
            print(f"truck y {truck_transform.location.y}")
            print(f"truck z {truck_transform.location.z}")
            print(f"distance {distance}")

        return distance

    def get_distance_to_waypoint_line(self,truck_transform, truck_forward_vector, waypoint_plus_current):
        # Obtaining two points on the truck forward vector
        # r = starting point + constant(direction)
        # r = (a.b) + t (x,y)
        t = 2
        truck_point_on_forward_vector_x = truck_transform.location.x + t*truck_forward_vector.x
        truck_point_on_forward_vector_y = truck_transform.location.y + t*truck_forward_vector.y

        truck_point_0 = Point(x=truck_transform.location.x,y=truck_transform.location.y)
        truck_point_1 = Point(x=truck_point_on_forward_vector_x,y=truck_point_on_forward_vector_y)

        # Obtaining two points perpendicular to the next waypoint
        waypoint_right_vector = self.route[self.last_waypoint_index + waypoint_plus_current].get_right_vector()
        # r = starting point + constant(direction)
        # r = (a.b) + t (x,y)
        t = 2
        waypoint_point_on_right_vector_x = self.route[self.last_waypoint_index + waypoint_plus_current].location.x + t*waypoint_right_vector.x
        waypoint_point_on_right_vector_y = self.route[self.last_waypoint_index + waypoint_plus_current].location.y + t*waypoint_right_vector.y

        waypoint_point_0 = Point(x=self.route[self.last_waypoint_index + waypoint_plus_current].location.x,y=self.route[self.last_waypoint_index + waypoint_plus_current].location.y)
        waypoint_point_1 = Point(x=waypoint_point_on_right_vector_x,y=waypoint_point_on_right_vector_y)

        # Finding the intersection point between the truck forward vector and the perpendicular to the waypoint
        intersection = lineLineIntersection(truck_point_0,truck_point_1,waypoint_point_0,waypoint_point_1)

        if (intersection.x == 10 ** 9 and intersection.y == 10 ** 9):
            # parallel lines
            distance_to_point_of_intersection = 10
        else:
            distance_to_point_of_intersection = math.sqrt((truck_transform.location.x-intersection.x)**2 + (truck_transform.location.y-intersection.y)**2)

        plot = False
        if plot:
            f = plt.figure()
            f.set_figwidth(6)
            f.set_figheight(6)

            x_values = [intersection.x, truck_transform.location.x, self.route[self.last_waypoint_index].location.x, self.route[self.last_waypoint_index+1].location.x, self.route[self.last_waypoint_index+2].location.x]
            y_values = [intersection.y, truck_transform.location.y, self.route[self.last_waypoint_index].location.y, self.route[self.last_waypoint_index+1].location.y, self.route[self.last_waypoint_index+2].location.y]
            x_min = min(x_values)
            x_max = max(x_values)

            y_min = min(y_values)
            y_max = max(y_values)
            buffer = 3
            #
            plt.xlim([x_min - buffer, x_max + buffer])
            plt.ylim([y_min - buffer, y_max + buffer])

            # print(f"x_pos {current_position.x} y_pos {current_position.y}")
            # print(f"x_last {previous_position.x} y_last {previous_position.y}")
            # print(f"x_next {next_position.x} y_next {next_position.y}")

            # plt.plot([intersection.x, next_position.x], [previous_position.y, next_position.y])
            # plotting the points
            plt.plot(intersection.x, intersection.y, marker="o", markersize=3, markeredgecolor="red",
                     markerfacecolor="red", label='Intersection Point')
            plt.plot(truck_transform.location.x, truck_transform.location.y, marker="o", markersize=3, markeredgecolor="black",
                     markerfacecolor="black", label='Current truck position')
            plt.plot(self.route[self.last_waypoint_index].location.x, self.route[self.last_waypoint_index].location.y, marker="o", markersize=3, markeredgecolor="blue",
                     markerfacecolor="blue", label='Current Waypoint')
            plt.plot(self.route[self.last_waypoint_index+1].location.x, self.route[self.last_waypoint_index+1].location.y, marker="o", markersize=3, markeredgecolor="green",
                     markerfacecolor="green", label='Next + 1 Waypoint')
            plt.plot(self.route[self.last_waypoint_index + 2].location.x,
                     self.route[self.last_waypoint_index + 2].location.y, marker="o", markersize=3,
                     markeredgecolor="purple",
                     markerfacecolor="purple", label='Next + 2  Waypoint')
            plt.gca().invert_yaxis()
            # val = update_next_waypoint(current_position.x,current_position.y,previous_position.x,previous_position.y,next_position.x,next_position.y)

            current_waypoint = self.route[self.last_waypoint_index].location
            next_waypoint = self.route[self.last_waypoint_index+1].location
            if next_waypoint.x - current_waypoint.x == 0:
                x = []
                y = []
                for a in range(-20, 20):
                    x.append(a)
                    y.append(current_waypoint.y)
                plt.plot(x, y)
            elif next_waypoint.y - current_waypoint.y == 0:
                x = []
                y = []
                for a in range(-20, 20):
                    x.append(current_waypoint.x)
                    y.append(a)
                plt.plot(x, y)

            else:
                gradOfPrevToNext = (next_waypoint.y - current_waypoint.y) / (next_waypoint.x - current_waypoint.x)
                gradOfPerpendicular = -1 / gradOfPrevToNext
                cOfPerpendicular = current_waypoint.y - gradOfPerpendicular * current_waypoint.x
                print(f"gradOfPerpendicular {gradOfPerpendicular}")
                print(f"cOfPerpendicular {cOfPerpendicular}")
                x = []
                y = []
                for a in range(-20, 200):
                    x.append(a)
                    y.append(gradOfPerpendicular * a + cOfPerpendicular)
                plt.plot(x, y, label="Perpendicular")
            leg = plt.legend(loc='upper right')
            # if in_front_of_waypoint == 0:
            #     print('POINT ON LINE')
            #     plt.title(f"Result = ONLINE - {angle}")
            # if in_front_of_waypoint == 1:
            #     plt.title(f"Result = FORWARD - {angle}")
            # if in_front_of_waypoint == -1:
            #     plt.title(f"Result = BACKWARD - {angle}")
            # print('--------------------')
            #
            # # naming the x-axis
            plt.xlabel('x - axis')
            # naming the y-axis
            plt.ylabel('y - axis')

            # function to show the plot
            plt.show()

        return distance_to_point_of_intersection

    def set_route(self,failed_entry_spawn_locations):
        self.route_points = []

        if self.one_after_the_other:
            self.last_chosen_route += 1

            if self.total_number_of_routes != -1:
                self.last_chosen_route = self.last_chosen_route % self.total_number_of_routes

        self.entry_spawn_point_index, self.exit_spawn_point_index, self.route_lane, self.last_roundabout_choice, self.total_number_of_routes = get_entry_exit_spawn_point_indices_2_lane(failed_entry_spawn_locations,self.last_roundabout_choice, self.last_chosen_route,map_name=self.map_name,is_testing=self.one_after_the_other)
        # key = str(self.entry_spawn_point_index) + " | " + str(self.exit_spawn_point_index)
        # if self.chosen_routes.get(key) is None:
        #     self.chosen_routes[key] = 1
        # else:
        #     self.chosen_routes[key] += 1
        #
        # for key,value in self.chosen_routes.items():
        #     print(f"{key} : {value}")
        # print('---------')


        entry_spawn_point = self.map.get_spawn_points()[self.entry_spawn_point_index]
        exit_spawn_point = self.map.get_spawn_points()[self.exit_spawn_point_index]

        # # Specify more than one starting point so the RL doesn't always start from the same position
        # spawn_point_no = random.choice([33, 28, 27, 17, 14, 11, 10, 5])
        # spawn_points = [self.map.get_spawn_points()[spawn_point_no]]

        # Obtaining the route information
        start_waypoint = self.map.get_waypoint(entry_spawn_point.location)
        end_waypoint = self.map.get_waypoint(exit_spawn_point.location)

        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location

        sampling_resolution = 2
        global_planner = GlobalRoutePlanner(self.map, sampling_resolution)

        route_waypoints = global_planner.trace_route(start_location, end_location)
        self.last_waypoint_index = 0
        self.route.clear()
        last_x = -1
        last_y = -1
        last_waypoint_transform = None

        for route_waypoint in route_waypoints:

            # Some waypoint may be duplicated
            # Checking and ignoring duplicated points
            if abs(last_x - round(route_waypoint[0].transform.location.x, 2)) < 0.4 and abs(last_y - round(
                    route_waypoint[0].transform.location.y, 2)) < 0.4:
                continue

            last_x = round(route_waypoint[0].transform.location.x, 2)
            last_y = round(route_waypoint[0].transform.location.y, 2)


            # self.route.append(carla.Transform(
            #     carla.Location(self.normalise_map_location(route_waypoint[0].transform.location.x, 'x'),
            #                    self.normalise_map_location(route_waypoint[0].transform.location.y, 'y'),
            #                    0),
            #     carla.Rotation(0, 0, 0)))
            if last_waypoint_transform is not None:
                # Ensuring that the next waypoint is in front of the previous
                if -1 == self.is_in_front_of_waypoint_from_vector(line_point=last_waypoint_transform,vector_line=last_waypoint_transform.get_right_vector(),in_front_point=route_waypoint[0].transform):
                    last_waypoint_transform = route_waypoint[0].transform

                    self.route.append(route_waypoint[0].transform)
                    self.route_points.append(
                        (route_waypoint[0].transform.location.x, route_waypoint[0].transform.location.y))
            else:
                last_waypoint_transform = route_waypoint[0].transform
                self.route.append(route_waypoint[0].transform)
                self.route_points.append(
                    (route_waypoint[0].transform.location.x, route_waypoint[0].transform.location.y))

        return self.entry_spawn_point_index, entry_spawn_point

    def reset_hero(self, hero_config):
        """This function resets / spawns the hero vehicle and its sensors"""

        if self.visualise_all_routes:
            visualise_all_routes(self.map,map_name=self.map_name)

        # Part 1: destroy all sensors (if necessary)
        self.sensor_interface_truck.destroy()
        if hero_config["truckTrailerCombo"]:
            # print("TRAILER PART 1/7")
            self.sensor_interface_trailer.destroy()

        self.world.tick()

        # Part 2: Spawn the ego vehicle
        # user_spawn_points = hero_config["spawn_points"]
        # if user_spawn_points:
        #     spawn_points = []
        #     for transform in user_spawn_points:
        #
        #         transform = [float(x) for x in transform.split(",")]
        #         if len(transform) == 3:
        #             location = carla.Location(
        #                 transform[0], transform[1], transform[2]
        #             )
        #             waypoint = self.map.get_waypoint(location)
        #             waypoint = waypoint.previous(random.uniform(0, 5))[0]
        #             transform = carla.Transform(
        #                 location, waypoint.transform.rotation
        #             )
        #         else:
        #             assert len(transform) == 6
        #             transform = carla.Transform(
        #                 carla.Location(transform[0], transform[1], transform[2]),
        #                 carla.Rotation(transform[4], transform[5], transform[3])
        #             )
        #         spawn_points.append(transform)
        # else:



        # print('ROUTE INFORMATION')
        # for route_waypoint in self.route:
        #     print(route_waypoint)

        # Where we generate the truck
        self.hero_blueprints = random.choice(get_actor_blueprints(self.world, hero_config["blueprintTruck"], "2"))
        self.hero_blueprints.set_attribute("role_name", "hero")

        if hero_config["truckTrailerCombo"]:
            # print("TRAILER PART 2/7")
            self.trailer_blueprints = random.choice(get_actor_blueprints(self.world, hero_config["blueprintTrailer"], "2"))
            self.trailer_blueprints.set_attribute("role_name", "hero-trailer")

        # If already spawned, destroy it
        if self.hero is not None:
            self.hero.destroy()
            self.hero = None

        if hero_config["truckTrailerCombo"] and self.hero_trailer is not None:
            # print("TRAILER PART 3/7")
            self.hero_trailer.destroy()
            self.hero_trailer = None


        # random.shuffle(spawn_points, random.random)
        # # for i in range(0,len(spawn_points)):
        # next_spawn_point = spawn_points[i % len(spawn_points)]
        failed_entry_spawn_locations = [-1]
        # while self.hero is None and (hero_config["truckTrailerCombo"] and self.hero_trailer is None) :

        while ((self.hero is None) or (self.hero_trailer is None)) if hero_config["truckTrailerCombo"] else (self.hero is None):

            entry_spawn_point_index, entry_spawn_point = self.set_route(failed_entry_spawn_locations)

            if hero_config["truckTrailerCombo"]:
                # print("TRAILER PART 4/7")
                # Spawning the trailer first and than spawning the truck in a location a bit forward up to connect with it
                entry_spawn_point.location.z = 0.5
                self.hero_trailer = self.world.try_spawn_actor(self.trailer_blueprints, entry_spawn_point)

            # Moving the spawn point a bit further up
            entry_spawn_point.location.z = 0.5
            forwardVector = entry_spawn_point.get_forward_vector() * 5.2
            entry_spawn_point.location += forwardVector

            # Spawning the truck
            self.hero = self.world.try_spawn_actor(self.hero_blueprints, entry_spawn_point)

            if self.hero is not None:
                print("Truck spawned!")
                if hero_config["truckTrailerCombo"]:
                    # print("TRAILER PART 5/7")
                    if self.hero_trailer is not None:
                        print("Trailer spawned!")
                    else:
                        print('FAILED TO SPAWN TRAILER')

            else:
                failed_entry_spawn_locations.append(entry_spawn_point_index)
                print("Could not spawn hero, changing spawn point")
                print('====> IF ERRORING HERE CHECK CODE in carla_core when generating spawn_points <====')

        if self.hero is None:
            print("We ran out of spawn points")
            print('====> IF ERRORING HERE CHECK CODE in carla_core when generating spawn_points<====')
            return
        if hero_config["truckTrailerCombo"] and self.hero_trailer is None:
            # print("TRAILER PART 6/7")
            print("We ran out of spawn points")
            print('====> IF ERRORING HERE CHECK CODE in carla_core when generating spawn_points<====')
            return



        self.world.tick()

        # Part 3: Spawn the new sensors
        # Where we set the sensors
        for name, attributes in hero_config["sensors"].items():
            if 'lidar_trailer' not in name:
                sensor_truck = SensorFactory.spawn(name, attributes, self.sensor_interface_truck, self.hero)
                # time.sleep(0.15)
            if hero_config["truckTrailerCombo"] and (name == 'collision' or 'lidar_trailer' in name):
                # print("TRAILER PART 7/7")
                # time.sleep(0.15)
                sensor_trailer = SensorFactory.spawn(name, attributes, self.sensor_interface_trailer, self.hero_trailer)
        # time.sleep(0.15)
        # Not needed anymore. This tick will happen when calling CarlaCore.tick()
        # self.world.tick()

        return self.hero

    def spawn_npcs(self, n_vehicles, n_walkers):
        """Spawns vehicles and walkers, also setting up the Traffic Manager and its parameters"""

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # Spawn vehicles
        spawn_points = self.world.get_map().get_spawn_points()
        n_spawn_points = len(spawn_points)

        if n_vehicles < n_spawn_points:
            random.shuffle(spawn_points)
        elif n_vehicles > n_spawn_points:
            logging.warning("{} vehicles were requested, but there were only {} available spawn points"
                            .format(n_vehicles, n_spawn_points))
            n_vehicles = n_spawn_points

        v_batch = []
        v_blueprints = self.world.get_blueprint_library().filter("vehicle.*")

        for n, transform in enumerate(spawn_points):
            if n >= n_vehicles:
                break
            v_blueprint = random.choice(v_blueprints)
            if v_blueprint.has_attribute('color'):
                color = random.choice(v_blueprint.get_attribute('color').recommended_values)
                v_blueprint.set_attribute('color', color)
            v_blueprint.set_attribute('role_name', 'autopilot')

            transform.location.z += 1
            v_batch.append(SpawnActor(v_blueprint, transform)
                           .then(SetAutopilot(FutureActor, True, self.tm_port)))

        results = self.client.apply_batch_sync(v_batch, True)
        if len(results) < n_vehicles:
            logging.warning("{} vehicles were requested but could only spawn {}"
                            .format(n_vehicles, len(results)))
        vehicles_id_list = [r.actor_id for r in results if not r.error]

        # Spawn the walkers
        spawn_locations = [self.world.get_random_location_from_navigation() for i in range(n_walkers)]

        w_batch = []
        w_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")

        for spawn_location in spawn_locations:
            w_blueprint = random.choice(w_blueprints)
            if w_blueprint.has_attribute('is_invincible'):
                w_blueprint.set_attribute('is_invincible', 'false')
            w_batch.append(SpawnActor(w_blueprint, carla.Transform(spawn_location)))

        results = self.client.apply_batch_sync(w_batch, True)
        if len(results) < n_walkers:
            logging.warning("Could only spawn {} out of the {} requested walkers."
                            .format(len(results), n_walkers))
        walkers_id_list = [r.actor_id for r in results if not r.error]

        # Spawn the walker controllers
        wc_batch = []
        wc_blueprint = self.world.get_blueprint_library().find('controller.ai.walker')

        for walker_id in walkers_id_list:
            wc_batch.append(SpawnActor(wc_blueprint, carla.Transform(), walker_id))

        results = self.client.apply_batch_sync(wc_batch, True)
        if len(results) < len(walkers_id_list):
            logging.warning("Only {} out of {} controllers could be created. Some walkers might be stopped"
                            .format(len(results), n_walkers))
        controllers_id_list = [r.actor_id for r in results if not r.error]

        self.world.tick()

        for controller in self.world.get_actors(controllers_id_list):
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())

        self.world.tick()
        self.actors = self.world.get_actors(vehicles_id_list + walkers_id_list + controllers_id_list)

    def tick(self, control):
        """Performs one tick of the simulation, moving all actors, and getting the sensor data"""
        # import time

        # Move hero vehicle
        if control is not None:
            self.apply_hero_control(control)

        # Tick once the simulation
        self.world.tick()
        # Move the spectator
        if self.custom_enable_rendering:
            self.set_spectator_camera_view()

        # start = time.time()
        # Return the new sensor data
        # x = self.get_sensor_data()
        # stop = time.time()
        # self.times_crazy.append(stop-start)
        # print(f'AVERAGE HERE22222222222222 {sum(self.times_crazy)/len(self.times_crazy)}')
        # return x
        return self.get_sensor_data()

    def set_spectator_camera_view(self):
        """This positions the spectator as a 3rd person view of the hero vehicle"""
        transform = self.hero.get_transform()



        # Get the camera position
        server_view_x = transform.location.x - 11 * transform.get_forward_vector().x
        server_view_y = transform.location.y - 11 * transform.get_forward_vector().y
        server_view_z = transform.location.z + 22
        # For car
        # server_view_x = transform.location.x - 5 * transform.get_forward_vector().x
        # server_view_y = transform.location.y - 5 * transform.get_forward_vector().y
        # server_view_z = transform.location.z + 3

        # For truck
        # server_view_x = transform.location.x - 11 * transform.get_forward_vector().x
        # server_view_y = transform.location.y - 11 * transform.get_forward_vector().y
        # server_view_z = transform.location.z + 18


        # Get the camera orientation
        server_view_roll = transform.rotation.roll
        server_view_yaw = transform.rotation.yaw
        server_view_pitch = transform.rotation.pitch - 75
        # For car transform.rotation.pitch
        # For truck transform.rotation.pitch - 75

        # Get the spectator and place it on the desired position
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(pitch=server_view_pitch, yaw=server_view_yaw, roll=server_view_roll),
            )
        )

    def apply_hero_control(self, control):
        """Applies the control calcualted at the experiment to the hero"""
        self.hero.apply_control(control)

    def get_sensor_data(self):
        """Returns the data sent by the different sensors at this tick"""
        sensor_data_truck = self.sensor_interface_truck.get_data('truck')
        if self.sensor_interface_trailer != None:
            sensor_data_trailer = self.sensor_interface_trailer.get_data('trailer')
        # print("---------")
        # world_frame = self.world.get_snapshot().frame
        # print("World frame: {}".format(world_frame))
        # for name, data in sensor_data.items():
        #     print("{}: {}".format(name, data[0]))

        if self.sensor_interface_trailer != None:
            return {**sensor_data_truck, **sensor_data_trailer}
        else:
            return sensor_data_truck

    # def dot(self,v, w):
    #     x, y, z = v
    #     X, Y, Z = w
    #     return x * X + y * Y + z * Z
    #
    # def length(self,v):
    #     x, y, z = v
    #     return math.sqrt(x * x + y * y + z * z)
    #
    # def vector(self,b, e):
    #     x, y, z = b
    #     X, Y, Z = e
    #     return (X - x, Y - y, Z - z)
    #
    # def unit(self,v):
    #     x, y, z = v
    #     mag = self.length(v)
    #     return (x / mag, y / mag, z / mag)
    #
    # def distance(self,p0, p1):
    #     return self.length(self.vector(p0, p1))
    #
    # def scale(self,v, sc):
    #     x, y, z = v
    #     return (x * sc, y * sc, z * sc)
    #
    # def add(self,v, w):
    #     x, y, z = v
    #     X, Y, Z = w
    #     return (x + X, y + Y, z + Z)
    #
    # # Given a line with coordinates 'start' and 'end' and the
    # # coordinates of a point 'pnt' the proc returns the shortest
    # # distance from pnt to the line and the coordinates of the
    # # nearest point on the line.
    # #
    # # 1  Convert the line segment to a vector ('line_vec').
    # # 2  Create a vector connecting start to pnt ('pnt_vec').
    # # 3  Find the length of the line vector ('line_len').
    # # 4  Convert line_vec to a unit vector ('line_unitvec').
    # # 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
    # # 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
    # # 7  Ensure t is in the range 0 to 1.
    # # 8  Use t to get the nearest location on the line to the end
    # #    of vector pnt_vec_scaled ('nearest').
    # # 9  Calculate the distance from nearest to pnt_vec_scaled.
    # # 10 Translate nearest back to the start/end line.
    # # Malcolm Kesson 16 Dec 2012
    # # https://stackoverflow.com/questions/27161533/find-the-shortest-distance-between-a-point-and-line-segments-not-line
    # def pnt2line(self,truck_transform,waypoint_plus_current):
    #     waypoint_right_vector = self.route[self.last_waypoint_index + waypoint_plus_current].get_right_vector()
    #     if self.route_lane == "left":
    #         # Truck should be on LEFT lane. free RIGHT lane
    #         t_left = -1.85
    #         t_right = 5.55
    #
    #     elif self.route_lane == "right":
    #         # Truck should be on RIGHT lane. free LEFT lane
    #         t_left = -5.55
    #         t_right = 1.85
    #     else:
    #         raise Exception('NO LANE')
    #
    #     left_point_x = self.route[
    #                        self.last_waypoint_index + waypoint_plus_current].location.x + t_left * waypoint_right_vector.x
    #     left_point_y = self.route[
    #                        self.last_waypoint_index + waypoint_plus_current].location.y + t_left * waypoint_right_vector.y
    #     right_point_x = self.route[
    #                         self.last_waypoint_index + waypoint_plus_current].location.x + t_right * waypoint_right_vector.x
    #     right_point_y = self.route[
    #                         self.last_waypoint_index + waypoint_plus_current].location.y + t_right * waypoint_right_vector.y
    #
    #
    #     start = (left_point_x, left_point_y, 0)
    #     end = (right_point_x, right_point_y, 0)
    #     pnt = (truck_transform.location.x, truck_transform.location.y, 0)
    #
    #     line_vec = self.vector(start, end)
    #     pnt_vec = self.vector(start, pnt)
    #     line_len = self.length(line_vec)
    #     line_unitvec = self.unit(line_vec)
    #     pnt_vec_scaled = self.scale(pnt_vec, 1.0 / line_len)
    #     t = self.dot(line_unitvec, pnt_vec_scaled)
    #     if t < 0.0:
    #         t = 0.0
    #     elif t > 1.0:
    #         t = 1.0
    #     nearest = self.scale(line_vec, t)
    #     dist = self.distance(nearest, pnt_vec)
    #     nearest = self.add(nearest, start)
    #     return (dist, nearest)

    # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment


    def sqr(self,x):
        return x ** 2

    def dist2(self,v, w):
        return self.sqr(v.x - w.x) + self.sqr(v.y - w.y)

    def distToSegmentSquared(self,p, v, w):
        # Calculates closest distance from point p to line formed by v -> w
        l2 = self.dist2(v, w)
        if l2 == 0:
            return self.dist2(p, v)
        t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2
        t = max(0, min(1, t))

        return self.dist2(p, Vector(x=v.x + t * (w.x - v.x), y=v.y + t * (w.y - v.y)))

    def distToSegment(self,truck_transform,waypoint_plus_current):
        p = Vector(x=truck_transform.location.x, y=truck_transform.location.y)

        waypoint_right_vector = self.route[self.last_waypoint_index + waypoint_plus_current].get_right_vector()
        if self.route_lane == "left":
            # Truck should be on LEFT lane. free RIGHT lane
            t_left = -1.85
            t_right = 5.55

        elif self.route_lane == "right":
            # Truck should be on RIGHT lane. free LEFT lane
            t_left = -5.55
            t_right = 1.85
        else:
            raise Exception('NO LANE')

        left_point_x = self.route[
                           self.last_waypoint_index + waypoint_plus_current].location.x + t_left * waypoint_right_vector.x
        left_point_y = self.route[
                           self.last_waypoint_index + waypoint_plus_current].location.y + t_left * waypoint_right_vector.y
        right_point_x = self.route[
                            self.last_waypoint_index + waypoint_plus_current].location.x + t_right * waypoint_right_vector.x
        right_point_y = self.route[
                            self.last_waypoint_index + waypoint_plus_current].location.y + t_right * waypoint_right_vector.y


        v = Vector(x=left_point_x, y=left_point_y)
        w = Vector(x=right_point_x, y=right_point_y)
        return math.sqrt(self.distToSegmentSquared(p, v, w))

    def shortest_distance_to_center_of_lane(self, truck_transform):
        p = Vector(x=truck_transform.location.x, y=truck_transform.location.y)

        previous_waypoint_location = self.route[self.last_waypoint_index - 1].location
        current_waypoint_location = self.route[self.last_waypoint_index].location


        v = Vector(x=previous_waypoint_location.x,y=previous_waypoint_location.y)
        w = Vector(x= current_waypoint_location.x,y=current_waypoint_location.y)


        return math.sqrt(self.distToSegmentSquared(p, v, w))