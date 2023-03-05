from typing import Tuple
from dataclasses import dataclass

@dataclass
class SweeperConfig:
    """Configuration for the sweeper speed."""
    num_max_steps           : int                 = 1000        # number of steps before the episode ends
    observation_type        : str                 = 'dict'      # 'dict', 'torch-no-grid', 'torch-grid'
    action_type             : str                 = 'continuous'# 'continuous', 'discrete-minimum', 'discrete' or 'discrete-x' with x an integer
    acceleration_range      : Tuple[float, float] = (-60, 60)   # units/s**2
    velocity_range          : Tuple[float, float] = (-6, 12)    # units/s
    steering_angle_range    : Tuple[float, float] = (-200, 200) # degrees/s
    radar_max_distance      : float               = 10          # units
    friction                : float               = 5.0         # 1/s
    sweeper_size            : Tuple[float, float] = (2., 1.)    # units
    num_radars              : int                 = 16          # number of rays

    def scale(self, resolution):
        self.acceleration_range = (self.acceleration_range[0] * resolution, self.acceleration_range[1] * resolution)
        self.velocity_range = (self.velocity_range[0] * resolution, self.velocity_range[1] * resolution)
        self.radar_max_distance *= resolution
        self.sweeper_size = (self.sweeper_size[0] * resolution, self.sweeper_size[1] * resolution)


@dataclass
class RewardConfig:
    """Configuration for the reward function."""
    factor_area_cleaned     : float = 1.0
    penalty_collision       : float = -100.0
    penalty_per_second      : float = -0.1
    done_on_collision       : bool  = False


@dataclass
class SweeperStats:
    """Statistics for the sweeper."""
    time                    : float = 0.0
    last_reward             : float = 0.0
    total_reward            : float = 0.0
    area_empty              : float = 0.0
    area_cleaned            : float = 0.0
    percentage_cleaned      : float = 0.0
    collisions              : int = 0

    def update(self, new_area: float, new_reward: float, had_collision: bool, dt: float):
        self.time += dt
        self.last_reward = new_reward
        self.total_reward += new_reward
        self.area_cleaned += new_area
        self.percentage_cleaned = 100 * self.area_cleaned / self.area_empty
        if had_collision:
            self.collisions += 1


@dataclass
class RenderOptions:
    """Render options for the sweeper game."""

    # General
    render                  : bool = True
    width                   : int = 800
    height                  : int = 800
    show_sweeper            : bool = True
    simulation_speed        : float = 1.0
    first_render            : bool = True

    # Infos
    show_fps                : bool = False
    show_time               : bool = False
    show_score              : bool = False
    show_simulation_speed   : bool = False

    # Path
    show_path       : bool = False
    path_color      : Tuple[int, int, int] = (0, 0, 255)
    path_alpha_decay : float = 0.1
    path_num_points : int = 100

    # Bounding bow and center
    show_bounding_box : bool = False
    bounding_box_color : Tuple[int, int, int] = (255, 0, 0)
    show_sweeper_center      : bool = False
    sweeper_center_color     : Tuple[int, int, int] = (255, 0, 0)

    # Map
    show_map        : bool = True
    cell_size       : int = 16
    cell_empty_color : Tuple[int, int, int] = (255, 255, 255)
    cell_obstacle_color : Tuple[int, int, int] = (0, 0, 0)
    cell_cleaned_color : Tuple[int, int, int] = (0, 255, 0)
    show_area       : bool = True

    # Velocity
    show_velocity   : bool = False
    velocity_color  : Tuple[int, int, int] = (0, 0, 255)

    # Distance sensor
    show_distance_sensors : bool = False
    distance_sensors_color : Tuple[int, int, int] = (255, 0, 255)