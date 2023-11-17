#!/usr/bin/env python
import math
# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import queue
from datetime import datetime
import open3d as o3d
import numpy as np
import time, sys, pickle
from datetime import datetime
import pandas as pd
from PIL import Image
from matplotlib import cm

from Helper import save_to_pickle, append_to_txt

LABEL_COLORS = np.array([
    (255, 255, 255),  # None
    (70, 70, 70),  # Building
    (100, 40, 40),  # Fences
    (55, 90, 80),  # Other
    (220, 20, 60),  # Pedestrian
    (153, 153, 153),  # Pole
    (157, 234, 50),  # RoadLines
    (255, 0, 4),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),  # Vehicle
    (102, 102, 156),  # Wall
    (220, 220, 0),  # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),  # Ground
    (150, 100, 100),  # Bridge
    (230, 150, 140),  # RailTrack
    (180, 165, 180),  # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160),  # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),  # Water
    (145, 170, 100),  # Terrain
]) / 255.0  # normalize each channel [0-1] since is what Open3D uses


class SensorInterface(object):
    """
    Class used to handle all the sensor data management
    """

    def __init__(self):
        self._sensors = {}  # {name: Sensor object}
        self._data_buffers = queue.Queue()
        self._queue_timeout = 60

        self._event_sensors = {}
        self._event_data_buffers = queue.Queue()

        self.visualiseLIDAR = False
        self.visualiseRADAR = False

        if self.visualiseRADAR and self.visualiseLIDAR:
            raise Exception('Cannot visualise both LIDAR and RADAR at the same time')

        self.visualiseCamera = False
        self.counter = 0
        self.lidar_window()

        self.VIRIDIS = np.array(cm.get_cmap('plasma').colors)
        self.COOL_RANGE = np.linspace(0.0, 1.0, self.VIRIDIS.shape[0])
        self.COOL = np.array(cm.get_cmap('winter')(self.COOL_RANGE))
        self.COOL = self.COOL[:, :3]



    def lidar_window(self):
        if self.visualiseLIDAR or self.visualiseRADAR:
            name = 'lidar' if self.visualiseLIDAR else 'radar'
            self.lidar_point_list = o3d.geometry.PointCloud()
            self.radar_point_list = o3d.geometry.PointCloud()

            self.vis = o3d.visualization.Visualizer()

            self.vis.create_window(
                window_name=name,
                width=960,
                height=540,
                left=480,
                top=270)

            axis = o3d.geometry.LineSet()
            axis.points = o3d.utility.Vector3dVector(np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]]))
            axis.lines = o3d.utility.Vector2iVector(np.array([
                [0, 1],
                [0, 2],
                [0, 3]]))
            axis.colors = o3d.utility.Vector3dVector(np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]]))
            self.vis.add_geometry(axis)
            self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            self.vis.get_render_option().point_size = 1
            self.vis.get_render_option().show_coordinate_frame = True

            self.frame = 0
            self.dt0 = datetime.now()

    @property
    def sensors(self):
        sensors = self._sensors.copy()
        sensors.update(self._event_sensors)
        return sensors

    def destroy(self):
        for sensor in self.sensors.values():
            sensor.destroy()
        self._data_buffers = queue.Queue()
        self._event_data_buffers = queue.Queue()

    def register(self, name, sensor):
        """Adds a specific sensor to the class"""
        if sensor.is_event_sensor():
            self._event_sensors[name] = sensor
        else:
            self._sensors[name] = sensor

    def get_data(self,blueprintName, carla_pid):
        """Returns the data of all the registered sensors as a dictionary {sensor_name: sensor_data}"""
        try:
            data_dict = {}
            while len(data_dict.keys()) < len(self._sensors.keys()):
                sensor_data = self._data_buffers.get(True, self._queue_timeout)

                if self.counter > 20 and 'lidar' in sensor_data[0] and self.visualiseLIDAR:
                    data = sensor_data[2]
                    # print(f"Number of LIDAR points {len(data)}")

                    # We're negating the y to correclty visualize a world that matches
                    # what we see in Unreal since Open3D uses a right-handed coordinate system
                    points = np.array([data[:,0], -data[:,1], data[:,2]]).T
                    # points = np.array([data['x'], -data['y'], data['z']]).T

                    # # An example of adding some noise to our data if needed:
                    # points += np.random.uniform(-0.05, 0.05, size=points.shape)

                    # Colorize the pointcloud based on the CityScapes color palette
                    labels = data[:,4].astype('uint32')
                    # labels = np.array(data['ObjTag'])
                    # import collections
                    # print(f"COUNTER: {collections.Counter(labels)}")

                    int_color = LABEL_COLORS[labels]

                    # # In case you want to make the color intensity depending
                    # # of the incident ray angle, you can use:
                    # int_color *= np.array(data['CosAngle'])[:, None]

                    self.lidar_point_list.points = o3d.utility.Vector3dVector(points)
                    self.lidar_point_list.colors = o3d.utility.Vector3dVector(int_color)

                    if self.frame == 2:
                        self.vis.add_geometry(self.lidar_point_list)
                    self.vis.update_geometry(self.lidar_point_list)

                    self.vis.poll_events()
                    self.vis.update_renderer()
                    # # This can fix Open3D jittering issues:
                    time.sleep(0.005)

                    process_time = datetime.now() - self.dt0
                    sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
                    sys.stdout.flush()
                    self.dt0 = datetime.now()
                    self.frame += 1

                if self.counter > 20 and 'radar' in sensor_data[0] and self.visualiseRADAR:
                    data = sensor_data[2]

                    radar_data = np.zeros((len(data), 4))

                    print(f'LEN RADAR DATA {len(data)}')
                    print(f'DATA {data}')

                    for i, detection in enumerate(data):
                        x = detection[2] * math.cos(detection[0]) * math.cos(detection[1])
                        y = detection[2] * math.cos(detection[0]) * math.sin(detection[1])
                        z = detection[2] * math.sin(detection[0])

                        radar_data[i, :] = [x, y, z, detection[3]]

                    intensity = np.abs(radar_data[:, -1])
                    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
                    int_color = np.c_[
                        np.interp(intensity_col, self.COOL_RANGE, self.COOL[:, 0]),
                        np.interp(intensity_col, self.COOL_RANGE, self.COOL[:, 1]),
                        np.interp(intensity_col, self.COOL_RANGE, self.COOL[:, 2])]

                    points = radar_data[:, :-1]
                    points[:, :1] = -points[:, :1]
                    print(f'POINTS {points}')
                    print(f'int_color {int_color}')
                    self.radar_point_list.points = o3d.utility.Vector3dVector(points)
                    self.radar_point_list.colors = o3d.utility.Vector3dVector(int_color)

                    if self.frame == 2:
                        self.vis.add_geometry(self.radar_point_list)
                    self.vis.update_geometry(self.radar_point_list)

                    self.vis.poll_events()
                    self.vis.update_renderer()
                    # # This can fix Open3D jittering issues:
                    time.sleep(0.005)

                    process_time = datetime.now() - self.dt0
                    sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
                    sys.stdout.flush()
                    self.dt0 = datetime.now()
                    self.frame += 1


                if self.visualiseCamera and (sensor_data[0] == 'semantic_camera' or sensor_data[0] == "depth_camera"):

                    # current_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S%Z")

                    img = Image.fromarray(sensor_data[2],None)
                    img.show()
                    time.sleep(0.005)
                    img.close()
                    # sensor_data[2].save_to_disk("image" + str(current_time),carla.ColorConverter.CityScapesPalette)

                data_dict[sensor_data[0]+'_'+blueprintName] = (sensor_data[1], sensor_data[2])
                self.counter+=1
        except queue.Empty:
            append_to_txt('failed_pids', str(datetime.now()) + ' ---- ' + str(carla_pid) + '\n')
            raise RuntimeError("A sensor took too long to send their data")

        for event_sensor in self._event_sensors:
            try:
                sensor_data = self._event_data_buffers.get_nowait()
                data_dict[sensor_data[0]+'_'+blueprintName] = (sensor_data[1], sensor_data[2])
            except queue.Empty:
                pass
        # print("Data buffer: {}".format(len(self._data_buffers.queue)))
        # print("Event data buffer: {}".format(len(self._event_data_buffers.queue)))
        return data_dict
