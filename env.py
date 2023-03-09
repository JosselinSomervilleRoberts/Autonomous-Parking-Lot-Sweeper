import gym
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import pygame
from map import Map
from config import SweeperConfig, RewardConfig, RenderOptions, SweeperStats
import torch
import configparser

screen_config = configparser.ConfigParser()
screen_config.read('screen_config.ini')

def draw_transparent_circle(screen, radius, color, alpha, center):
    x, y = center
    surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA) # create a surface with an alpha channel
    pygame.draw.circle(surface, color, (radius, radius), radius) # draw the circle onto the surface
    surface.set_alpha(alpha) # set the alpha value of the surface
    screen.blit(surface, (x - radius, y - radius)) # draw the surface onto the screen at the adjusted position


def between(minimum, maximum):
    def func(val):
        return minimum <= val <= maximum
    return func

def float_to_str(vale, nb_char_before_comma=3, nb_char_after_comma=4):
    """If the number of character before the comma is not enough, some spaces are added
    If the number of character after the comma is not enough, some 0 are added
    If the number of character after the comma is too much, the number is rounded"""
    rounded = str(round(vale, nb_char_after_comma))
    # Add a comma if there is none
    if '.' not in rounded:
        rounded += '.'
    # Add spaces before the comma
    if len(rounded.split('.')[0]) < nb_char_before_comma:
        rounded = ' ' * (nb_char_before_comma - len(rounded.split('.')[0])) + rounded
    # Add 0 after the comma
    if len(rounded.split('.')[1]) < nb_char_after_comma:
        rounded += '0' * (nb_char_after_comma - len(rounded.split('.')[1]))
    return rounded

def get_aligned_text_and_value(text, value, nb_char_text=20, nb_char_before_comma=3, nb_char_after_comma=4):
    """Return a string with the text and the value aligned"""
    # Pads the text
    text_padded = text + ' ' * (nb_char_text - len(text))
    # Pads the value
    value = float_to_str(value, nb_char_before_comma=nb_char_before_comma, nb_char_after_comma=nb_char_after_comma)
    return text_padded + ": " + value


class Sweeper:
    def __init__(self, sweeper_config: SweeperConfig):
        self.conf = sweeper_config

        # Linear
        self.position = np.array([0., 0.], dtype=np.float32)
        self.speed = 0.
        self.acceleration = 0.
        self.friction = sweeper_config.friction

        # Rotation
        self.angle = 0.
        self.angle_speed = 0.

        # Vehicle size
        self.size = np.array([sweeper_config.sweeper_size[0], sweeper_config.sweeper_size[1]])

    def update_position(self, acceleration, steering, dt=None):
        if dt is None: dt = 1. / self.conf.action_frequency
        # Set acceleration and steering
        self.acceleration = max(min(acceleration, self.conf.acceleration_range[1]), self.conf.acceleration_range[0])
        self.angle_speed = max(min(steering, self.conf.steering_angle_range[1]), self.conf.steering_angle_range[0])

        # Update position and angle (Explicit Euler)
        old_speed = self.speed
        self.speed += self.acceleration * dt
        self.speed *= (1 - self.friction * dt)
        self.speed = max(min(self.speed, self.conf.speed_range[1]), self.conf.speed_range[0])
        # The speed can't change sign
        if self.speed * old_speed < 0:
            self.speed = 0
        if abs(self.speed) < 0.1: self.speed = 0.
        self.angle += self.angle_speed * dt
        angle = self.angle * np.pi / 180.
        self.position += self.speed * dt * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    def get_bounding_box(self, factor = 1.):
        # Get bounding box of the sweeper (the position is the center of the sweeper)
        x, y = self.position
        w, h = self.size * factor
        angle = self.angle * np.pi / 180.
        return np.array([
            [x + w/2 * np.cos(angle) - h/2 * np.sin(angle), y + w/2 * np.sin(angle) + h/2 * np.cos(angle)],
            [x - w/2 * np.cos(angle) - h/2 * np.sin(angle), y - w/2 * np.sin(angle) + h/2 * np.cos(angle)],
            [x - w/2 * np.cos(angle) + h/2 * np.sin(angle), y - w/2 * np.sin(angle) - h/2 * np.cos(angle)],
            [x + w/2 * np.cos(angle) + h/2 * np.sin(angle), y + w/2 * np.sin(angle) - h/2 * np.cos(angle)]
        ])

    def get_upper_bounding_box(self, factor = 1.):
        # Get upper bounding box of the sweeper (the position is the center of the sweeper)
        x, y = self.position
        w, h = self.size * factor
        angle = self.angle * np.pi / 180.
        return np.array([
            [x + w/2 * np.cos(angle) - h/2 * np.sin(angle), y + w/2 * np.sin(angle) + h/2 * np.cos(angle)],
            [x                       - h/2 * np.sin(angle), y                       + h/2 * np.cos(angle)],
            [x                       + h/2 * np.sin(angle), y                       - h/2 * np.cos(angle)],
            [x + w/2 * np.cos(angle) + h/2 * np.sin(angle), y + w/2 * np.sin(angle) - h/2 * np.cos(angle)]
        ])

    def get_lower_bounding_box(self, factor = 1.):
        # Get lower bounding box of the sweeper (the position is the center of the sweeper)
        x, y = self.position
        w, h = self.size * factor
        angle = self.angle * np.pi / 180.
        return np.array([
            [x                       - h/2 * np.sin(angle), y                       + h/2 * np.cos(angle)],
            [x - w/2 * np.cos(angle) - h/2 * np.sin(angle), y - w/2 * np.sin(angle) + h/2 * np.cos(angle)],
            [x - w/2 * np.cos(angle) + h/2 * np.sin(angle), y - w/2 * np.sin(angle) - h/2 * np.cos(angle)],
            [x                       + h/2 * np.sin(angle), y                       - h/2 * np.cos(angle)]
        ])

    def get_front_position(self):
        angle = self.angle * np.pi / 180.
        return self.position + self.size[0] * 0.5 * np.array([np.cos(angle), np.sin(angle)])


class Radar:

    def __init__(self, sweeper_config: SweeperConfig, i: int, resolution: int, cell_value: int, radius:int, max_distance: int, n_radars: int):
        self.i = i
        self.resolution = resolution
        self.radius = radius
        self.max_distance = max_distance
        self.cell_value = cell_value
        self.n_radars = n_radars

        # Position and angle of the radar
        self.rel_pos = np.array([0., 0.])
        self.angle = - 90 + 360 * i / n_radars

        if i >= n_radars and cell_value == Map.CELL_OBSTACLE:
            i -= n_radars
            # The first 3 radars are on the front of the sweeper pointing forward
            # The 4th and 5th are on the front of the sweeper pointing left and right
            # The 6th and 7th are on the back of the sweeper pointing left and right
            # The 8th, 9th and 10th are on the back of the sweeper pointing backward
            # The 11th and 12th are on the left and right side of the sweeper outside
            MARGIN = 0.05
            if i < 3:
                self.angle = 0
                self.rel_pos[0] = sweeper_config.sweeper_size[0] * 0.5
                if i ==1: # Left
                    self.rel_pos[1] = -sweeper_config.sweeper_size[1] * (0.5 + MARGIN)
                elif i==2: # Right
                    self.rel_pos[1] = sweeper_config.sweeper_size[1] * (0.5 + MARGIN)
            elif i == 3:
                self.angle = -90
                self.rel_pos[0] = sweeper_config.sweeper_size[0] * (0.5 + MARGIN)
                self.rel_pos[1] = -sweeper_config.sweeper_size[1] * 0.5
            elif i == 4:
                self.angle = 90
                self.rel_pos[0] = sweeper_config.sweeper_size[0] * (0.5 + MARGIN)
                self.rel_pos[1] = sweeper_config.sweeper_size[1] * 0.5
            elif i == 5:
                self.angle = -90
                self.rel_pos[0] = -sweeper_config.sweeper_size[0] * (0.5 + MARGIN)
                self.rel_pos[1] = -sweeper_config.sweeper_size[1] * 0.5
            elif i == 6:
                self.angle = 90
                self.rel_pos[0] = -sweeper_config.sweeper_size[0] * (0.5 + MARGIN)
                self.rel_pos[1] = sweeper_config.sweeper_size[1] * 0.5
            elif i < 10:
                self.angle = 180
                self.rel_pos[0] = -sweeper_config.sweeper_size[0] * 0.5
                if i == 8:
                    self.rel_pos[1] = -sweeper_config.sweeper_size[1] * (0.5 + MARGIN)
                elif i == 9:
                    self.rel_pos[1] = sweeper_config.sweeper_size[1] * (0.5 + MARGIN)
            else:
                raise Exception("Invalid additional obstacle rador of radius 0 index (from 0 to 9 included)")


    def get_rad_angle(self, sweeper: Sweeper):
        return (sweeper.angle + self.angle) * np.pi / 180.

    def get_position(self, sweeper: Sweeper):
        # Get the position of the radar
        pos = sweeper.position.copy()
        sweeper_rad_angle = np.deg2rad(sweeper.angle)
        cos_sweeper_angle = np.cos(sweeper_rad_angle)
        sin_sweeper_angle = np.sin(sweeper_rad_angle)
        pos[0] += self.rel_pos[0] * cos_sweeper_angle - self.rel_pos[1] * sin_sweeper_angle
        pos[1] += self.rel_pos[0] * sin_sweeper_angle + self.rel_pos[1] * cos_sweeper_angle
        return pos
        
    def measure(self, sweeper: Sweeper, map: Map):
        # Get the position of the radar
        rad_angle = self.get_rad_angle(sweeper)
        position = self.get_position(sweeper)

        # Compute the distance
        self.distance = map.compute_distance_to_closest_zone_of_value(pos=position, rad_angle=rad_angle, value=self.cell_value, radius=self.radius,min_ratio=0.9, max_distance=self.max_distance, set_minus_one_on_out_of_bounds=self.cell_value != Map.CELL_OBSTACLE, min_distance=0.0 + 3.0*self.resolution * (self.cell_value == Map.CELL_CLEANED))
        if self.distance >= 0: self.distance /= self.resolution * self.max_distance
        return self.distance


class AdvancedRadar:

    def __init__(self, i: int, sweeper_config: SweeperConfig, resolution: float = 1, max_radius: float = 1, n_radars: int = 1):
        self.i = i
        self.resolution = resolution
        self.max_radius = max_radius
        self.max_distances = sweeper_config.radar_max_distance.copy()
        self.n_radars = n_radars

        # Position and angle of the radar
        self.rel_pos = np.array([0., 0.])
        self.angle = - 90 + 360 * i / n_radars

        self._distances = - np.ones((3, max_radius + 1), dtype=np.float32)
        self._norm_distances = - np.ones((3, max_radius + 1), dtype=np.float32)


    def get_rad_angle(self, sweeper: Sweeper):
        return (sweeper.angle + self.angle) * np.pi / 180.

    def get_position(self, sweeper: Sweeper):
        # Get the position of the radar
        pos = sweeper.position.copy()
        sweeper_rad_angle = np.deg2rad(sweeper.angle)
        cos_sweeper_angle = np.cos(sweeper_rad_angle)
        sin_sweeper_angle = np.sin(sweeper_rad_angle)
        pos[0] += self.rel_pos[0] * cos_sweeper_angle - self.rel_pos[1] * sin_sweeper_angle
        pos[1] += self.rel_pos[0] * sin_sweeper_angle + self.rel_pos[1] * cos_sweeper_angle
        return pos

    def measure(self, sweeper: Sweeper, map: Map):
        # Get the position of the radar
        rad_angle = self.get_rad_angle(sweeper)
        position = self.get_position(sweeper)

        # Compute the distance
        # We compute in the order (empty, cleaned, obstacle) as it is
        # more efficient in ters of skipping (see compute_distance_to_closest_zone_of_value_multi_radius_multi_value,
        # the use of the min_ratio >= 0.5 trick).
        self._distances = map.compute_distance_to_closest_zone_of_value_multi_radius_multi_value(pos=position, rad_angle=rad_angle, values=[Map.CELL_EMPTY, Map.CELL_CLEANED, Map.CELL_OBSTACLE], max_radius=self.max_radius, max_distances=self.max_distances, resolution=self.resolution)
        
        # Reorder the results
        self._distances = np.array([self._distances[0], self._distances[2], self._distances[1]])

        # Normalize the distances
        idx = np.where(self._distances >= 0)
        self._norm_distances = self._distances.copy()
        self._norm_distances[idx] /= self.max_distances[idx]
        return self._norm_distances

    def get_normalized_distances(self) -> np.ndarray:
        return self._norm_distances

    def get_distances(self) -> np.ndarray:
        return self._distances

    def get_distances_obstacles(self) -> np.ndarray:
        return self._distances[2]

    def get_distances_cleaned(self) -> np.ndarray:
        return self._distances[1]

    def get_distances_empty(self) -> np.ndarray:
        return self._distances[0]

    def get_distance_to_closest_obstacle(self) -> float:
        return self._distances[1][0]

    
class SweeperEnv(gym.Env):

    ACTION_INDEX_ACCELERATION = 0
    ACTION_INDEX_STEERING = 1

    def __init__(self, sweeper_config: SweeperConfig, reward_config: RewardConfig, render_options: RenderOptions, resolution: float = 1, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.sweeper_config = sweeper_config
        self.render_options = render_options
        self.reward_config = reward_config
        self.resolution = resolution

        # Init everything
        sweeper_config.scale(resolution)
        self.init_map_and_sweeper(sweeper_config, resolution)
        self.init_gym(sweeper_config)

        # Reset the environment
        self.reset()

    def init_gym(self, sweeper_config: SweeperConfig) -> None:
        # Action Space:
        if sweeper_config.action_type == "continuous":
            # - Acceleration (continuous) # m/s^2
            # - Steering (continuous)     # degrees/s
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -1]), high=np.array([1, 1])
            )
        elif sweeper_config.action_type == "discrete-minimum":
            # - 0: accelerate max forward and don't turn
            # - 1: don't accelerate and don't turn
            # - 2: accelerate max backward and don't turn
            # - 3: accelerate max forward and turn max left
            # - 4: don't accelerate and turn max left
            # - 5: accelerate max backward and turn max left
            # - 6: accelerate max forward and turn max right
            # - 7: don't accelerate and turn max right
            # - 8: accelerate max backward and turn max right
            self.action_space = gym.spaces.Discrete(9)
        elif sweeper_config.action_type == "discrete" \
            or "-" in sweeper_config.action_type and sweeper_config.action_type.split("-")[0] == "discrete":
            nb_discretization = 10
            if "-" in sweeper_config.action_type:
                nb_discretization = int(sweeper_config.action_type.split("-")[1])
            # Discritize the action space in nb_discretization^2
            self.action_space = gym.spaces.Discrete(nb_discretization**2)
        elif sweeper_config.action_type == "multi-discrete" \
            or "-" in sweeper_config.action_type and sweeper_config.action_type.split("-")[0] == "multi-discrete":
            nb_discretization = 10
            if "-" in sweeper_config.action_type:
                nb_discretization = int(sweeper_config.action_type.split("-")[1])
            # Discritize the action space in nb_discretization^2
            self.action_space = gym.spaces.MultiDiscrete([nb_discretization, nb_discretization])
        else:
            raise Exception("Unknown action type: " + sweeper_config.action_type)
        self.action_to_acceleration_and_steering = self.get_action_to_acceleration_and_steering_fn()
        
        # State space:
        self.observation_space = None
        if sweeper_config.observation_type == "simple":
            # Gets only speed and radar distances
            self.observation_space = gym.spaces.Box(
                low=np.array([-1] + [-1] * (sweeper_config.num_radars)),
                high=np.array([1] + [1] * (sweeper_config.num_radars)),
                dtype=np.float32
            )
        elif sweeper_config.observation_type == "simple-double-radar":
            # Gets only speed and radar distances
            num_radars = 2 * sweeper_config.num_radars# + 10
            self.observation_space = gym.spaces.Box(
                low=np.array([-1] + [-1] * num_radars),
                high=np.array([1] + [1] * num_radars),
                dtype=np.float32
            )
        elif sweeper_config.observation_type == "simple-radar-cnn":
            n_cells = len(self.sweeper_config.radar_max_distance)
            n_r = len(self.sweeper_config.radar_max_distance[0])
            n_dir = self.sweeper_config.radar_max_distance[0][0]
            self.observation_space = gym.spaces.Dict({
                "radars": gym.spaces.Box(
                    low=-1, high=1, shape=(n_cells, n_dir, n_r), dtype=np.float32
                ),
                "other": gym.spaces.Box(
                    low=-1, high=1, shape=(1,), dtype=np.float32
                )
            })
        elif sweeper_config.observation_type == "grid-only":
            # Gets only the grid
            # Reshapes self.map.grid from (width, height) to (width, height, 1)
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.map.width, self.map.height, 1), dtype=Map.CELL_TYPE
            )
        elif sweeper_config.observation_type == "complex":
            # Gets the grid, the radar distances, the normalized position, the speed and the direction
            # Reshapes self.map.grid from (width, height) to (width, height, 1)
            n_cells = len(self.sweeper_config.radar_max_distance)
            n_r = len(self.sweeper_config.radar_max_distance[0])
            n_dir = self.sweeper_config.radar_max_distance[0][0]
            num_radars = n_cells * n_dir * n_r
            self.observation_space = gym.spaces.Dict({
                "grid": gym.spaces.Box(
                    low=0, high=255, shape=(self.map.width, self.map.height, 1), dtype=Map.CELL_TYPE
                ),
                "radars": gym.spaces.Box(
                    low=-1, high=1, shape=(num_radars,), dtype=np.float32
                ),
                "position": gym.spaces.Box(
                    low=0, high=1, shape=(2,), dtype=np.float32
                ),
                "speed": gym.spaces.Box(
                    low=-1, high=1, shape=(1,), dtype=np.float32
                ),
                "direction": gym.spaces.Box(
                    low=-1, high=1, shape=(2,), dtype=np.float32
                ),
            })
        else:
            raise Exception("Unknown observation type: " + sweeper_config.observation_type)



    def init_map_and_sweeper(self, sweeper_config: SweeperConfig, resolution: float = 1, new_map: bool = True, generate_new_map: bool = False) -> None:
        GAME_WIDTH, GAME_HEIGHT = 50, 50
        self.render_options.cell_size = int(round(int(screen_config['SCREEN']['DEFAULT_CELL_SIZE']) / resolution))
        self.render_options.first_render = True
        map_width = int(round(GAME_WIDTH * resolution))
        map_height = int(round(GAME_HEIGHT * resolution))
        self.render_options.width = map_width * self.render_options.cell_size
        self.render_options.height = map_height * self.render_options.cell_size

        # Create map
        if new_map:
            self.map = Map(map_width, map_height, self.render_options)
            if generate_new_map:
                self.map.init_random()
            else:
                self.map.load_random()
        else:
            self.map.clear()
        self.map.compute_matrix_cum(max_radius=sweeper_config.radar_max_radius)

        # Create sweeper
        self.sweeper = Sweeper(sweeper_config)

    def _get_grid_for_observation(self):
        return self.map.get_reshaped_grid_with_sweeper(self.sweeper)

    def _get_observation(self):
        if self.sweeper_config.observation_type == 'simple':
            # Old radars
            # radar_values = np.array([radar.distance for radar in self.radars[self.map.i_cum[Map.CELL_OBSTACLE]][0]])

            # Advanced radars
            radar_values = np.array([radar.get_distance_to_closest_obstacle() for radar in self.radars])

            return np.array([
                self.sweeper.speed / self.sweeper_config.speed_range[1],
                *radar_values
            ], dtype=np.float32)
        elif self.sweeper_config.observation_type == 'simple-double-radar':
            # Old radars
            # radar_values = []
            # for r2 in self.radars:
            #     for r1 in r2:
            #         for radar in r1:
            #             radar_values.append(radar.distance)
            # radar_values = np.array(radar_values, dtype=np.float32)

            # Advanced radars
            radar_values = []
            for radar in self.radars:
                radar_values += radar.get_normalized_distances().copy().flatten().tolist()
            radar_values = np.array(radar_values, dtype=np.float32)

            return np.array([
                self.sweeper.speed / self.sweeper_config.speed_range[1],
                *radar_values
            ], dtype=np.float32)
        elif self.sweeper_config.observation_type == 'simple-radar-cnn':
            # Old radars
            # # Returns a dict with a 3D array for the radar distances and a 1D array for the other values
            # n_cells = len(self.radars)
            # n_r = len(self.radars[0])
            # n_dir = len(self.radars[0][0])
            # rad_3D = -np.ones((n_cells, n_dir, n_r), dtype=np.float32)
            # for idx_cell, radars_for_cell in enumerate(self.radars):
            #     for idx_r, radars_for_cell_for_r in enumerate(radars_for_cell):
            #         for idx_dir, radar in enumerate(radars_for_cell_for_r):
            #             rad_3D[idx_cell, idx_dir, idx_r] = radar.distance

            # Advanced radars
            rad_3D = np.array([radar.get_normalized_distances() for radar in self.radars], dtype=np.float32) # shape (n_dir, n_cells, n_r)
            rad_3D.swapaxes(0, 1) # shape (n_cells, n_dir, n_r)

        elif self.sweeper_config.observation_type == 'grid-only':
            return self._get_grid_for_observation()
        elif self.sweeper_config.observation_type == 'complex':
            # Old radars
            # radar_values = []
            # for r2 in self.radars:
            #     for r1 in r2:
            #         for radar in r1:
            #             radar_values.append(radar.distance)
            # radar_values = np.array(radar_values, dtype=np.float32)

            # Advanced radars
            radar_values = []
            for radar in self.radars:
                radar_values += radar.get_normalized_distances().copy().flatten().tolist()
            radar_values = np.array(radar_values, dtype=np.float32)

            return {
                "grid": self._get_grid_for_observation(),
                "radars": radar_values,
                "position": self.sweeper.position.copy() / np.array([self.map.width, self.map.height], dtype=np.float32),
                "speed": np.array([self.sweeper.speed / self.sweeper_config.speed_range[1]], dtype=np.float32),
                "direction": np.array([np.cos(np.deg2rad(self.sweeper.angle)), np.sin(np.deg2rad(self.sweeper.angle))], dtype=np.float32)
            }
        else:
            raise Exception("Unknown observation type: " + self.sweeper_config.observation_type)

    def get_action_to_acceleration_and_steering_fn(self):
        if self.sweeper_config.action_type == "continuous":
            return lambda action: (
                np.interp(action[0], [-1, 1], self.sweeper_config.acceleration_range),
                np.interp(action[1], [-1, 1], self.sweeper_config.steering_angle_range)
            )
        elif self.sweeper_config.action_type == "discrete-minimum":
            FACTOR = 0.75
            return lambda action: {
                0: (FACTOR * self.sweeper_config.acceleration_range[1], 0),
                1: (0, 0),
                2: (FACTOR * self.sweeper_config.acceleration_range[0], 0),
                3: (FACTOR * self.sweeper_config.acceleration_range[1], FACTOR * self.sweeper_config.steering_angle_range[0]),
                4: (0, FACTOR * self.sweeper_config.steering_angle_range[0]),
                5: (FACTOR * self.sweeper_config.acceleration_range[0], FACTOR * self.sweeper_config.steering_angle_range[0]),
                6: (FACTOR * self.sweeper_config.acceleration_range[1], FACTOR * self.sweeper_config.steering_angle_range[1]),
                7: (0, FACTOR * self.sweeper_config.steering_angle_range[1]),
                8: (FACTOR * self.sweeper_config.acceleration_range[0], FACTOR * self.sweeper_config.steering_angle_range[1]),
            }[action]
        elif self.sweeper_config.action_type == "discrete" \
            or "-" in self.sweeper_config.action_type and self.sweeper_config.action_type.split("-")[0] == "discrete":
            nb_discretization = 10
            if "-" in self.sweeper_config.action_type:
                nb_discretization = int(self.sweeper_config.action_type.split("-")[1])
            # Discritize the action space in nb_discretization^2
            return lambda action: (
                self.sweeper_config.acceleration_range[0] + (self.sweeper_config.acceleration_range[1] - self.sweeper_config.acceleration_range[0]) * action // nb_discretization,
                self.sweeper_config.steering_angle_range[0] + (self.sweeper_config.steering_angle_range[1] - self.sweeper_config.steering_angle_range[0]) * (action % nb_discretization)
            )
        elif sweeper_config.action_type == "multi-discrete" \
            or "-" in self.sweeper_config.action_type and self.sweeper_config.action_type.split("-")[0] == "multi-discrete":
            nb_discretization = 10
            if "-" in self.sweeper_config.action_type:
                nb_discretization = int(self.sweeper_config.action_type.split("-")[1])
            # Discritize the action space in nb_discretization^2
            return lambda action: (
                self.sweeper_config.acceleration_range[0] + (self.sweeper_config.acceleration_range[1] - self.sweeper_config.acceleration_range[0]) * action[0] // nb_discretization,
                self.sweeper_config.steering_angle_range[0] + (self.sweeper_config.steering_angle_range[1] - self.sweeper_config.steering_angle_range[0]) * action[1] // nb_discretization
            )


    def step(self, action, dt=None):
        """Returns (observation, reward, terminated, info)"""
        if dt is None: dt = 1. / self.sweeper_config.action_frequency
        dt *= self.render_options.simulation_speed

        # Update sweeper
        self.iter += 1
        (acceleration, steering) = self.action_to_acceleration_and_steering(action)
        prev_position = self.sweeper.position.copy()
        prev_angle = self.sweeper.angle

        self.sweeper.update_position(acceleration, steering, dt)

        # Check for collision
        # If there is one, find with a binary search the last position where there was no collision
        had_collision = False
        if self.check_collision():
            had_collision = True
            collision_position = self.sweeper.position.copy()
            if self.debug: print("Collision at", collision_position, "from", prev_position, "with action", action)
            # Binary search between prev_position and collision_position
            N_BINARY_SEARCH = 3
            fmin, fmax = 0., 1.
            for _ in range(N_BINARY_SEARCH):
                fmid = (fmin + fmax) / 2.
                self.sweeper.position = prev_position + fmid * (collision_position - prev_position)
                self.sweeper.angle = prev_angle + fmid * (self.sweeper.angle - prev_angle)
                if self.check_collision():
                    fmax = fmid
                else:
                    fmin = fmid
            STEP_BACK_ON_COLLISION = 0.5
            norm_dir = np.linalg.norm(collision_position - prev_position)
            normalized_dir = (collision_position - prev_position) / norm_dir
            self.sweeper.position = prev_position + (fmin * norm_dir - STEP_BACK_ON_COLLISION) * normalized_dir
            # Check if self.sweeper.position cointains a nan
            if np.isnan(self.sweeper.position).any():
                self.sweeper.position = prev_position
            self.sweeper.angle = prev_angle + fmin * (self.sweeper.angle - prev_angle)
            self.sweeper.speed = 0.
            self.sweeper.acceleration = 0.

        # Compute reward
        # Position of front of sweeper
        front_position = self.sweeper.get_front_position()
        new_area = self.map.apply_cleaning(front_position, width=2.6*self.sweeper.size[1], resolution=self.resolution)
        self.sweeper_positions.append([self.sweeper.position[0], self.sweeper.position[1]])
        factor = (1. + self.stats.percentage_cleaned / 100.) ** 2
        reward = (3./7) * factor * self.reward_config.reward_area_total * new_area / self.stats.area_empty
        if abs(acceleration) < 0.05 * self.sweeper_config.acceleration_range[1]:
            reward += self.reward_config.reward_idle
        # (3./7) comes fron the integral of (1+x)^2 over [0,1] (factor)
        reward += self.reward_config.reward_per_second * dt + self.reward_config.reward_per_step
        if had_collision:
            reward += self.reward_config.reward_collision
        if self.sweeper.speed < 0:
            reward += self.reward_config.reward_backwards

        # Radars
        if np.linalg.norm(prev_position - self.sweeper.position) > 0.1 or abs(prev_angle - self.sweeper.angle) > 3:
            self.compute_radars()

        # Update stats
        self.stats.update(new_area=new_area, new_reward=reward, had_collision=had_collision, dt=dt)

        # Return observation, reward, terminated, truncated, info
        done = had_collision and self.reward_config.done_on_collision or self.stats.percentage_cleaned >= 90.0 or self.iter >= self.sweeper_config.num_max_steps
        return self._get_observation(), reward, done, {"collision": had_collision}

    def compute_radars(self):
        """Compute the values of the radars"""
        # Old radars
        # for idx, list_radar in enumerate(self.radars):
        #     for l_radar in list_radar:
        #         for radar in l_radar:
        #             radar.measure(self.sweeper, self.map)

        # Advanced radars
        for radar in self.radars:
            radar.measure(self.sweeper, self.map)

    def check_collision(self):
        return self.map.check_collision(self.sweeper.get_bounding_box())

    def reset(self, new_map=True, generate_new_map=False, *args):
        """Reset the environment and return an initial observation and info."""
        self.iter = 0
        self.stats = SweeperStats()

        # Reset map
        self.init_map_and_sweeper(sweeper_config=self.sweeper_config, resolution=self.resolution, new_map=new_map, generate_new_map=generate_new_map)
        self.map.cleaning_path = []
        self.stats.area_empty = self.map.get_empty_area(resolution=self.resolution)

        # Old radars
        # self.radars = []
        # for cell_value in self.map.i_cum.keys():
        #     self.radars.append([])
        #     idx_cell = self.map.i_cum[cell_value]
        #     for radius in range(len(self.sweeper_config.num_radars[idx_cell])):
        #         count = self.sweeper_config.num_radars[idx_cell][radius]
        #         if count == 0: continue
        #         max_distance = self.sweeper_config.radar_max_distance[idx_cell][radius]
        #         self.radars[-1].append([Radar(self.sweeper_config, i, resolution=self.resolution, cell_value=cell_value, radius=radius, max_distance=max_distance, n_radars=count) for i in range(count)])# + 10 * (radius==0 and cell_value==Map.CELL_OBSTACLE))])

        # Advanced radars
        n_radars = self.sweeper_config.num_radars
        max_radius = self.sweeper_config.radar_max_radius
        self.radars = [AdvancedRadar(i=i, sweeper_config=self.sweeper_config, resolution=self.resolution, max_radius=max_radius, n_radars=n_radars) for i in range(n_radars)]
        
        # Reset sweeper by setting it's position to a random empty cell
        empty_cells = self.map.get_empty_tiles()
        collision = True
        while collision:
            self.sweeper.position = empty_cells[np.random.randint(len(empty_cells))].astype(np.float32)
            self.sweeper.angle = np.random.randint(360)
            collision = self.check_collision()
            self.compute_radars()
            # Old radars
            # for i, radar in enumerate(self.radars[self.map.i_cum[Map.CELL_OBSTACLE]][0]):
            #     if 0 <= radar.distance * radar.max_distance < self.sweeper_config.spawn_min_distance_to_wall:
            #         collision = True
            #         break

            # Advanced radars
            for radar in self.radars:
                if 0 <= radar.get_distance_to_closest_obstacle() < self.sweeper_config.spawn_min_distance_to_wall:
                    collision = True
                    break
        self.compute_radars()

        # Reset sweeper
        self.sweeper.speed = 0
        self.sweeper.acceleration = 0
        self.sweeper.angle_speed = 0
        self.sweeper_positions = []

        # Return observation
        return self._get_observation()



    # ==================== Pygame rendering ==================== #

    def init_pygame(self):

        # Init pygame
        pygame.init()
        pygame.display.set_caption('Sweeper Environment')
        self.footer_size = 100
        self.screen = pygame.display.set_mode((self.render_options.width, self.render_options.height + self.footer_size))
        screen_size = np.array(self.screen.get_size())
        self.screen_center = screen_size / 2

        # Load images
        self.carImg = pygame.transform.scale(pygame.image.load('assets/car.png'),
            (self.sweeper.size[0] * self.render_options.cell_size, self.sweeper.size[1] * self.render_options.cell_size))

    def fill_shapely_outline(self, input, color=(0,0,0)):
        x, y = input.xy
        outline = [[x[i], y[i]] for i in range(len(x))]
        pygame.draw.polygon(self.screen, color, self.render_options.cell_size * np.array(outline))

    def render(self, clock: pygame.time.Clock = None):
        # Init pygame if it has not been initialized yet
        if not pygame.get_init():
            self.init_pygame()

        # Render map
        rerender = self.render_options.first_render \
            or self.render_options.show_path or self.render_options.show_radars_obstacle>=0 or self.render_options.show_radars_empty>=0 \
                or self.render_options.show_radars_cleaned>=0
        self.map.display(self.sweeper, self.screen, rerender=rerender)
        self.render_options.first_render = False

        # Draw the sweeper's path (with alpha decreasing with time)
        if self.render_options.show_path:
            transparency = 1.0
            for i in range(min(self.render_options.path_num_points, len(self.sweeper_positions)) - 1):
                pygame.draw.line(self.screen, (self.render_options.path_color[0], self.render_options.path_color[1], self.render_options.path_color[2], int(transparency * 255)),
                                 self.render_options.cell_size * np.array(self.sweeper_positions[-i-1]),
                                 self.render_options.cell_size * np.array(self.sweeper_positions[-i-2]),
                                 width=2)
                transparency *= self.render_options.path_alpha_decay

        # Draw the sweeper (seeper_pos is the center of the sweeper)
        sweeper_pos = self.sweeper.position
        if self.render_options.show_sweeper:
            sweeper_angle = self.sweeper.angle
            carImg_temp = pygame.transform.rotate(self.carImg, -sweeper_angle)
            carImg_rotated_rect = carImg_temp.get_rect()
            carImg_rotated_rect.center = sweeper_pos * self.render_options.cell_size
            self.screen.blit(carImg_temp, carImg_rotated_rect)

        # Draw bounding box of sweeper (a np.array of 4 points)
        if self.render_options.show_bounding_box:
            sweeper_bbox = self.sweeper.get_bounding_box()
            pygame.draw.lines(self.screen, self.render_options.bounding_box_color, True, self.render_options.cell_size * sweeper_bbox, width=2)

        # Display a circle around the sweeper's center
        if self.render_options.show_sweeper_center:
            pygame.draw.circle(self.screen, self.render_options.sweeper_center_color, self.render_options.cell_size * sweeper_pos, 5)
            pygame.draw.circle(self.screen, self.render_options.sweeper_center_color, self.render_options.cell_size * self.sweeper.get_front_position(), 3)

        # Display velocity vector as an arrow
        if self.render_options.show_velocity:
            direction = 0.2 * self.sweeper.speed * np.array([np.cos(np.deg2rad(self.sweeper.angle)), np.sin(np.deg2rad(self.sweeper.angle))])
            pygame.draw.line(self.screen, self.render_options.velocity_color, self.render_options.cell_size * sweeper_pos, self.render_options.cell_size * (sweeper_pos + direction), width=2)
            
        # Dislay radars obstacles
        alpha_surface = pygame.Surface((self.render_options.width, self.render_options.height), pygame.SRCALPHA)
        displays = [self.render_options.show_radars_empty, self.render_options.show_radars_obstacle, self.render_options.show_radars_cleaned]
        radars_color = [self.render_options.radars_empty_color, self.render_options.radars_obstacle_color, self.render_options.radars_cleaned_color]
        
        # Old radars
        # for cell_value, idx in self.map.i_cum.items():
        #     r = displays[idx]
        #     if r >= 0:
        #         list_radars = self.radars[idx][r]
        #         if cell_value == Map.CELL_OBSTACLE:
        #             self.display_value("Obstacle radius: ", list_radars[0].radius, 10, -30, color=radars_color[idx])
        #         elif cell_value == Map.CELL_CLEANED:
        #             self.display_value("Cleaned radius: ", list_radars[0].radius, int(self.render_options.width * 0.37), -30, color=radars_color[idx])
        #         elif cell_value == Map.CELL_EMPTY:
        #             self.display_value("Empty radius: ", list_radars[0].radius, int(self.render_options.width * 0.74), -30, color=radars_color[idx])
        #         for radar in list_radars:
        #             distance = radar.distance
        #             if distance >= 0:
        #                 rad_angle = radar.get_rad_angle(self.sweeper)
        #                 direction = self.resolution * distance * np.array([np.cos(rad_angle), np.sin(rad_angle)]) * radar.max_distance
        #                 position = radar.get_position(self.sweeper)
        #                 pygame.draw.line(alpha_surface, radars_color[idx], self.render_options.cell_size * position, self.render_options.cell_size * (position + direction), width=2) 
        #                 draw_transparent_circle(self.screen, radius=int(max(0.5,radar.radius) * self.render_options.cell_size), color=radars_color[idx], center=self.render_options.cell_size * (position + direction), alpha=100)
        #                 #pygame.draw.circle(alpha_surface, radars_color[idx], self.render_options.cell_size * (position + direction), int(self.render_options.cell_size * max(0.5, radar.radius)))
        
        # Advanced radars
        for cell_value, idx in self.map.i_cum.items():
            r = displays[idx]
            if r >= 0:
                if cell_value == Map.CELL_OBSTACLE:
                    self.display_value("Obstacle radius: ", r, 10, -30, color=radars_color[idx])
                elif cell_value == Map.CELL_CLEANED:
                    self.display_value("Cleaned radius: ", r, int(self.render_options.width * 0.37), -30, color=radars_color[idx])
                elif cell_value == Map.CELL_EMPTY:
                    self.display_value("Empty radius: ", r, int(self.render_options.width * 0.74), -30, color=radars_color[idx])
                for radar in self.radars:
                    distance = radar.get_distances()[idx][r]
                    if distance >= 0:
                        rad_angle = radar.get_rad_angle(self.sweeper)
                        direction = self.resolution * distance * np.array([np.cos(rad_angle), np.sin(rad_angle)])
                        position = radar.get_position(self.sweeper)
                        pygame.draw.line(alpha_surface, radars_color[idx], self.render_options.cell_size * position, self.render_options.cell_size * (position + direction), width=2) 
                        draw_transparent_circle(self.screen, radius=int(max(0.5,r) * self.render_options.cell_size), color=radars_color[idx], center=self.render_options.cell_size * (position + direction), alpha=100)
                        #pygame.draw.circle(alpha_surface, radars_color[idx], self.render_options.cell_size * (position + direction), int(self.render_options.cell_size * max(0.5, radar.radius)))
                

        # Updates the screen
        alpha_surface.set_alpha(50)
        self.screen.blit(alpha_surface, (0, 0))
        self.display_footer(clock=clock)
        pygame.display.flip()

    def display_value(self, text, value, x, y, color=(0, 0, 0)):
        """Display a text with a value on the screen."""
        font = pygame.font.SysFont('Arial', 16)
        
        # Display text
        text_rendered = font.render(text, True, color)
        self.screen.blit(text_rendered, (x, self.render_options.height + y))

        # Display ":" between text and value
        text_rendered = font.render(':', True, color)
        self.screen.blit(text_rendered, (x + 100, self.render_options.height + y))

        # Display value
        text_rendered = font.render(float_to_str(value), True, color)
        self.screen.blit(text_rendered, (x + 120, self.render_options.height + y))

    def display_footer(self, clock: pygame.time.Clock = None):
        """Displays some infos in the footer of the screen:
        On the left there is the current iteration, reward, cumulated reward and simulation speed.
        On the left there is the current speed, acceleration and angle speed of the sweeper.
        In the middle there is the time, the are cleaned, the percentage of the map cleaned and the number of collisions.
        All these values are stored in the stats object.
        """

        # Fill footer
        self.screen.fill((255, 255, 255), (0, self.render_options.height, self.render_options.width, self.footer_size))

        # Display iteration, reward, cumulated reward and simulation speed on the left
        self.display_value('Iteration', self.iter, 10, 10, color=(0, 0, 0))
        self.display_value('Reward', self.stats.last_reward, 10, 30, color=(0, 0, 0))
        self.display_value('Cum. reward', self.stats.total_reward, 10, 50, color=(0, 0, 0))
        self.display_value('Sim. speed', self.render_options.simulation_speed, 10, 70, color=(0, 0, 0))

        # Display speed, acceleration and angle speed and pygame framerate on the right in dark blue
        self.display_value('Speed', self.sweeper.speed / self.resolution, int(self.render_options.width * 0.74), 10, color=(0, 0, 255))
        self.display_value('Acceleration', self.sweeper.acceleration / self.resolution, int(self.render_options.width * 0.74), 30, color=(0, 0, 255))
        self.display_value('Angle speed', self.sweeper.angle_speed, int(self.render_options.width * 0.74), 50, color=(0, 0, 255))

        # Display pygame framerate in dark green
        if isinstance(clock, pygame.time.Clock):
            self.display_value('FPS', clock.get_fps(), int(self.render_options.width * 0.74), 70, color=(0, 255, 0))

        # Display time, area cleaned, percentage of the map cleaned and number of collisions in the middle in dark red
        self.display_value('Time', self.stats.time, int(self.render_options.width * 0.37), 10, color=(255, 0, 0))
        self.display_value('Area / seconds', self.stats.area_cleaned / max(1e-3, self.stats.time), int(self.render_options.width * 0.37), 30, color=(255, 0, 0))
        self.display_value('Perc. cleaned', self.stats.percentage_cleaned, int(self.render_options.width * 0.37), 50, color=(255, 0, 0))
        self.display_value('Num. collisions', self.stats.collisions, int(self.render_options.width * 0.37), 70, color=(255, 0, 0))

    def set_render_options(self, render_options: RenderOptions):
        """Sets the render options."""
        render_options.first_render = True
        self.render_options = render_options
        self.map.render_options = render_options



    def process_pygame_event(self, event):
        # Checks for key just pressed
        # - H: displays help (to summarize the keys)
        # - R: resets the environment on the same map
        # - T: resets the environment on an other map
        # - Q: quits the program
        # - B: toggles the display of the bounding box
        # - P: toggles the display of the path
        # - S: saves the map
        # - A: toggles the display of the area
        # - V: toggles the display of the velocity
        # - D: toggles the display of the debug information
        # - W: toggles the display of the sensors to walls
        # - E: toggles the display of the sensors to empty cells
        # - C: toggles the display of the sensors to cleaned cells
        # - N: creates a new map
        # - +: increases the speed of the simulation
        # - -: decreases the speed of the simulation

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                sys.exit()
            if event.key == pygame.K_r:
                observation = self.reset(new_map=False, generate_new_map=False)
            if event.key == pygame.K_t:
                observation = self.reset(new_map=True, generate_new_map=False)
            if event.key == pygame.K_h:
                print("Help")
                print("----")
                print("Q: Quit")
                print("R: Reset on same map")
                print("T: Reset on a another map")
                print("H: Help")
                print("B: Toggle bounding box")
                print("P: Toggle path")
                print("S: Save the map")
                print("A: Toggle area")
                print("V: Toggle velocity")
                print("D: Toggle debug")
                print("W: Toggle sensors to walls")
                print("E: Toggle sensors to empty cells")
                print("C: Toggle sensors to cleaned cells")
                print("N: Generate New map")
                print("+: Increase speed")
                print("-: Decrease speed")
                print("Left/Right: Steer")
                print("Up/Down: Accelerate/Decelerate")

            if event.key == pygame.K_b:
                self.render_options.show_bounding_box = not self.render_options.show_bounding_box
                self.render_options.show_sweeper_center = not self.render_options.show_sweeper_center
            if event.key == pygame.K_p:
                self.render_options.show_path = not self.render_options.show_path
                self.render_options.first_render = True
            if event.key == pygame.K_s:
                self.map.save()
            if event.key == pygame.K_a:
                self.render_options.show_area = not self.render_options.show_area
                self.render_options.first_render = True
                self.map.generate_image()
            if event.key == pygame.K_v:
                self.render_options.show_velocity = not self.render_options.show_velocity
            if event.key == pygame.K_d:
                self.debug = not self.debug
            if event.key == pygame.K_w:
                self.render_options.show_radars_obstacle += 1
                if self.render_options.show_radars_obstacle > self.sweeper_config.radar_max_radius:
                    self.render_options.show_radars_obstacle = -1
                self.render_options.first_render = True
            if event.key == pygame.K_e:
                self.render_options.show_radars_empty += 1
                if self.render_options.show_radars_empty > self.sweeper_config.radar_max_radius:
                    self.render_options.show_radars_empty = -1
                self.render_options.first_render = True
            if event.key == pygame.K_c:
                self.render_options.show_radars_cleaned += 1
                if self.render_options.show_radars_cleaned > self.sweeper_config.radar_max_radius:
                    self.render_options.show_radars_cleaned = -1
                self.render_options.first_render = True
            if event.key == pygame.K_n:
                self.reset(new_map=True, generate_new_map=True)
            if event.key == pygame.K_EQUALS:
                self.render_options.simulation_speed *= 2.0
                print("Speed: " + str(self.render_options.simulation_speed))
            if event.key == pygame.K_MINUS:
                self.render_options.simulation_speed /= 2.0
                print("Speed: " + str(self.render_options.simulation_speed))



if __name__ == "__main__":
    import pygame
    import sys
    import time

    # Implements a random agent for our gym environment
    sweeper_config = SweeperConfig()
    env = SweeperEnv(sweeper_config=sweeper_config, reward_config=RewardConfig(), render_options=RenderOptions(), resolution = 2.0, debug=False)
    observation = env.reset(new_map=True, generate_new_map=False)
    print(env.action_space)
    rewards = []
    cum_rewards = []
    cum_reward = 0
    positions_list = []
    past_action = (0, 0)

    clock = pygame.time.Clock()
    last_time_sec = time.time()
    last_action_time_sec = time.time() - 1./ env.sweeper_config.action_frequency
    steering = 0
    acceleration = 0

    while True:

        # Edit map
        if pygame.get_init():
            left, middle, right = pygame.mouse.get_pressed()
            if left:
                # Get the position of the mouse
                mouse_pos = pygame.mouse.get_pos()
                # Get the cell position
                cell_pos = (mouse_pos[0] // env.render_options.cell_size, mouse_pos[1] // env.render_options.cell_size)
                env.map.set_cell_value(cell_pos[0], cell_pos[1], Map.CELL_OBSTACLE)
            if right:
                # Get the position of the mouse
                mouse_pos = pygame.mouse.get_pos()
                # Get the cell position
                cell_pos = (mouse_pos[0] // env.render_options.cell_size, mouse_pos[1] // env.render_options.cell_size)
                env.map.set_cell_value(cell_pos[0], cell_pos[1], Map.CELL_EMPTY)


            # Checks for event to close the window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                else:
                    env.process_pygame_event(event)

            # Checks for key pressed
            t = time.time()
            dt_action = t - last_action_time_sec
            if True:#dt_action > 1. / env.sweeper_config.action_frequency:
                steering = 0
                acceleration = 0
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    steering = -1
                if keys[pygame.K_RIGHT]:
                    steering = 1
                if keys[pygame.K_UP]:
                    acceleration = 1
                if keys[pygame.K_DOWN]:
                    acceleration = -1
                last_action_time_sec = t


        clock.tick(sweeper_config.action_frequency)
        action = (acceleration, steering)
        # TODO: Use a random action. This is just for the baseline model
        # action = env.action_space.sample()
        # FILTER = 1 - 0.03 * ACCEL_BY
        # action = (FILTER * past_action[0] + (1 - FILTER) * action[0], FILTER * past_action[1] + (1 - FILTER) * action[1])
        # past_action = action
        new_time_sec = time.time()
        dt = new_time_sec - last_time_sec
        last_time_sec = new_time_sec
        observation, reward, terminated, info = env.step(action, dt=dt)
        env.render(clock=clock)
        if terminated:
            observation = env.reset()
        # Keep tracks of rewards
        rewards.append(reward)
        cum_reward += reward
        cum_rewards.append(cum_reward)
        positions_list.append(observation)

    # Close the window
    pygame.quit()

    # Plot rewards
    plt.plot(rewards, label="Rewards")
    plt.plot(cum_rewards, label="Cumulative Rewards")
    plt.legend()
    plt.show()

    # Plot positions
    positions = np.array(positions_list)
    plt.plot(positions[:, 0], positions[:, 1], label="Positions")
    plt.legend()
    # plt.show()
