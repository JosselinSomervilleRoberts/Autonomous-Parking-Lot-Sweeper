from typing import Tuple
from dataclasses import dataclass

@dataclass
class SweeperConfig:
    """Configuration for the sweeper speed."""
    num_max_steps           : int                 = 9999        # number of steps before the episode ends
    observation_type        : str                 = 'simple'    # 'simple', 'grid-only', 'complex'
    action_type             : str                 = 'continuous'# 'continuous', 'discrete-minimum', 'discrete', 'multi-discrete'
    acceleration_range      : Tuple[float, float] = (-30, 30)   # units/s**2
    speed_range             : Tuple[float, float] = (-6, 12)    # units/s
    steering_angle_range    : Tuple[float, float] = (-200, 200) # degrees/s
    radar_max_distance      : float               = 10          # units
    friction                : float               = 3.0         # 1/s
    sweeper_size            : Tuple[float, float] = (2., 1.)    # units
    num_radars              : int                 = 32          # number of rays
    spawn_min_distance_to_wall : float            = 2.5         # units

    def scale(self, resolution):
        self.acceleration_range = (self.acceleration_range[0] * resolution, self.acceleration_range[1] * resolution)
        self.speed_range = (self.speed_range[0] * resolution, self.speed_range[1] * resolution)
        self.radar_max_distance *= resolution
        self.sweeper_size = (self.sweeper_size[0] * resolution, self.sweeper_size[1] * resolution)

    def __str__(self):
        return f"""SweeperConfig(
    num_max_steps              : {self.num_max_steps}
    observation_type           : {self.observation_type}
    action_type                : {self.action_type}
    acceleration_range         : {self.acceleration_range}
    speed_range                : {self.speed_range}
    steering_angle_range       : {self.steering_angle_range}
    radar_max_distance         : {self.radar_max_distance}
    friction                   : {self.friction}
    sweeper_size               : {self.sweeper_size}
    num_radars                 : {self.num_radars}
    spawn_min_distance_to_wall : {self.spawn_min_distance_to_wall}
)"""


@dataclass
class RewardConfig:
    """Configuration for the reward function."""
    reward_area_total       : float = 10000
    reward_collision        : float = -1000.0
    reward_per_second       : float = 0
    reward_per_step         : float = -0.1
    reward_backwards        : float = -1.0
    reward_idle             : float = -2.0
    done_on_collision       : bool  = False

    def __str__(self):
        return f"""RewardConfig(
    reward_area_total       : {self.reward_area_total}
    reward_collision        : {self.reward_collision}
    reward_per_second       : {self.reward_per_second}
    reward_per_step         : {self.reward_per_step}
    reward_backwards        : {self.reward_backwards}
    done_on_collision       : {self.done_on_collision}
)"""


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

    def __str__(self):
        return f"""SweeperStats(
    time                    : {self.time}
    last_reward             : {self.last_reward}
    total_reward            : {self.total_reward}
    area_empty              : {self.area_empty}
    area_cleaned            : {self.area_cleaned}
    percentage_cleaned      : {self.percentage_cleaned}
    collisions              : {self.collisions}
)"""


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

    def __str__(self):
        return f"""RenderOptions(
    render                  : {self.render}
    width                   : {self.width}
    height                  : {self.height}
    show_sweeper            : {self.show_sweeper}
    simulation_speed        : {self.simulation_speed}
    first_render            : {self.first_render}
    show_fps                : {self.show_fps}
    show_time               : {self.show_time}
    show_score              : {self.show_score}
    show_simulation_speed   : {self.show_simulation_speed}
    show_path               : {self.show_path}
    path_color              : {self.path_color}
    path_alpha_decay        : {self.path_alpha_decay}
    path_num_points         : {self.path_num_points}
    show_bounding_box       : {self.show_bounding_box}
    bounding_box_color      : {self.bounding_box_color}
    show_sweeper_center     : {self.show_sweeper_center}
    sweeper_center_color    : {self.sweeper_center_color}
    show_map                : {self.show_map}
    cell_size               : {self.cell_size}
    cell_empty_color        : {self.cell_empty_color}
    cell_obstacle_color     : {self.cell_obstacle_color}
    cell_cleaned_color      : {self.cell_cleaned_color}
    show_area               : {self.show_area}
    show_velocity           : {self.show_velocity}
    velocity_color          : {self.velocity_color}
    show_distance_sensors   : {self.show_distance_sensors}
    distance_sensors_color  : {self.distance_sensors_color}
)"""