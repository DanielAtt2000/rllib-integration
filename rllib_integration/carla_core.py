#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import random
import signal
import time
import psutil
import logging

import datetime
import carla

from rllib_integration.RouteGeneration.global_route_planner import GlobalRoutePlanner
from rllib_integration.sensors.sensor_interface import SensorInterface
from rllib_integration.sensors.factory import SensorFactory
from rllib_integration.helper import join_dicts
from rllib_integration.GetStartStopLocation import get_entry_exit_spawn_point_indices

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
        self.sensor_interface_trailer = SensorInterface() if self.config["truckTrailerCombo"] else None
        self.server_port = 2000
        self.server_port_lines = ''

        self.route = []
        self.last_waypoint_index = None

        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None

        self.current_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        os.mkdir(f"results\\run_{self.current_time}")

        # self.init_server()
        self.connect_client()

    # def init_server(self):
    #     with open('../ppo_example/server_ports.txt','r') as portsFileRead:
    #         self.server_port_lines = portsFileRead.readlines()

    #     self.server_port = int(self.server_port_lines[0])

    #     with open('../ppo_example/server_ports.txt', 'w') as portsFileWrite:
    #         for line in self.server_port_lines:
    #             if str(self.server_port) not in line:
    #                 portsFileWrite.write(line)

    # def init_server(self):
    #     """Start a server on a random port"""
    #     self.server_port = random.randint(15000, 32000)
    #
    #     # Ray tends to start all processes simultaneously. Use random delays to avoid problems
    #     time.sleep(random.uniform(0, 1))
    #
    #     uses_server_port = is_used(self.server_port)
    #     uses_stream_port = is_used(self.server_port + 1)
    #     while uses_server_port and uses_stream_port:
    #         if uses_server_port:
    #             print("Is using the server port: " + self.server_port)
    #         if uses_stream_port:
    #             print("Is using the streaming port: " + str(self.server_port+1))
    #         self.server_port += 2
    #         uses_server_port = is_used(self.server_port)
    #         uses_stream_port = is_used(self.server_port+1)
    #
    #     if self.config["show_display"]:
    #         server_command = [
    #             "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
    #             "-windowed",
    #             "-ResX={}".format(self.config["resolution_x"]),
    #             "-ResY={}".format(self.config["resolution_y"]),
    #         ]
    #     else:
    #         server_command = [
    #             "DISPLAY= ",
    #             "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
    #             "-opengl"  # no-display isn't supported for Unreal 4.24 with vulkan
    #         ]
    #
    #     server_command += [
    #         "--carla-rpc-port={}".format(self.server_port),
    #         "-quality-level={}".format(self.config["quality_level"])
    #     ]
    #
    #     server_command_text = " ".join(map(str, server_command))
    #     print(server_command_text)
    #     server_process = subprocess.Popen(
    #         server_command_text,
    #         shell=True,
    #         preexec_fn=os.setsid,
    #         stdout=open(os.devnull, "w"),
    #     )

    def connect_client(self):
        """Connect to the client"""

        for i in range(self.config["retries_on_error"]):
            try:
                self.client = carla.Client(self.config["host"], self.server_port)
                self.client.set_timeout(self.config["timeout"])
                self.world = self.client.get_world()

                settings = self.world.get_settings()
                settings.no_rendering_mode = not self.config["enable_rendering"]
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.config["timestep"]
                self.world.apply_settings(settings)
                self.world.tick()

                return

            except Exception as e:
                print(" Waiting for server to be ready: {}, attempt {} of {}".format(e, i + 1, self.config["retries_on_error"]))
                time.sleep(3)

        raise Exception("Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration")

    def setup_experiment(self, experiment_config):
        """Initialize the hero and sensors"""

        self.world = self.client.load_world(
            map_name=experiment_config["town"],
            reset_settings=False,
            # map_layers = carla.MapLayer.All if self.config["enable_map_assets"] else carla.MapLayer.NONE
            map_layers=carla.MapLayer.NONE
        )

        self.map = self.world.get_map()

        # Choose the weather of the simulation
        weather = getattr(carla.WeatherParameters, experiment_config["weather"])
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

    def set_route(self,failed_entry_spawn_locations):
        entry_spawn_point_index, exit_spawn_point_index = get_entry_exit_spawn_point_indices(failed_entry_spawn_locations)
        entry_spawn_point = self.map.get_spawn_points()[entry_spawn_point_index]
        exit_spawn_point = self.map.get_spawn_points()[exit_spawn_point_index]

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
        last_x = -1
        last_y = -1

        for route_waypoint in route_waypoints:

            # Some waypoint may be duplicated
            # Checking and ignoring duplicated points
            if last_x == round(route_waypoint[0].transform.location.x, 5) and last_y == round(
                    route_waypoint[0].transform.location.y, 5):
                continue

            last_x = round(route_waypoint[0].transform.location.x, 5)
            last_y = round(route_waypoint[0].transform.location.y, 5)

            self.route.append(carla.Transform(
                carla.Location(self.normalise_map_location(route_waypoint[0].transform.location.x, 'x'),
                               self.normalise_map_location(route_waypoint[0].transform.location.y, 'y'),
                               0),
                carla.Rotation(0, 0, 0)))

        return entry_spawn_point_index, entry_spawn_point

    def reset_hero(self, hero_config):
        """This function resets / spawns the hero vehicle and its sensors"""

        # Part 1: destroy all sensors (if necessary)
        self.sensor_interface_truck.destroy()
        if hero_config["truckTrailerCombo"]:
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
            self.trailer_blueprints = random.choice(get_actor_blueprints(self.world, hero_config["blueprintTrailer"], "2"))
            self.trailer_blueprints.set_attribute("role_name", "hero-trailer")

        # If already spawned, destroy it
        if self.hero is not None:
            self.hero.destroy()
            self.hero = None

        if hero_config["truckTrailerCombo"] and self.hero_trailer is not None:
            self.hero_trailer.destroy()
            self.hero_trailer = None


        # random.shuffle(spawn_points, random.random)
        # # for i in range(0,len(spawn_points)):
        # next_spawn_point = spawn_points[i % len(spawn_points)]
        failed_entry_spawn_locations = [-1]
        print(f'self.hero {self.hero}')
        # while self.hero is None and (hero_config["truckTrailerCombo"] and self.hero_trailer is None) :
        while self.hero is None:

            entry_spawn_point_index, entry_spawn_point = self.set_route(failed_entry_spawn_locations)
            print(f'SPAWN POINT FOUND AT {entry_spawn_point_index}')

            if hero_config["truckTrailerCombo"]:
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
            print("We ran out of spawn points")
            print('====> IF ERRORING HERE CHECK CODE in carla_core when generating spawn_points<====')
            return



        self.world.tick()

        # Part 3: Spawn the new sensors
        # Where we set the sensors
        for name, attributes in hero_config["sensors"].items():
            sensor_truck = SensorFactory.spawn(name, attributes, self.sensor_interface_truck, self.hero)
            if hero_config["truckTrailerCombo"] and name != 'lidar':
                sensor_trailer = SensorFactory.spawn(name, attributes, self.sensor_interface_trailer, self.hero_trailer)

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

        # Move hero vehicle
        if control is not None:
            self.apply_hero_control(control)

        # Tick once the simulation
        self.world.tick()
        # Move the spectator
        if self.config["enable_rendering"]:
            self.set_spectator_camera_view()

        print(f'SELFHER IN TICK {self.hero}')
        # Return the new sensor data
        return self.get_sensor_data()

    def set_spectator_camera_view(self):
        """This positions the spectator as a 3rd person view of the hero vehicle"""
        transform = self.hero.get_transform()

        print(f"hero transform {transform}")

        # Get the camera position
        server_view_x = transform.location.x - 5 * transform.get_forward_vector().x
        server_view_y = transform.location.y - 5 * transform.get_forward_vector().y
        server_view_z = transform.location.z + 3

        # Get the camera orientation
        server_view_roll = transform.rotation.roll
        server_view_yaw = transform.rotation.yaw
        server_view_pitch = transform.rotation.pitch

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
