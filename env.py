import gymnasium as gym
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
        self.position = np.array([0., 0.])
        self.speed = 0.
        self.acceleration = 0.
        self.friction = sweeper_config.friction

        # Rotation
        self.angle = 0.
        self.angle_speed = 0.

        # Vehicle size
        self.size = np.array([sweeper_config.sweeper_size[0], sweeper_config.sweeper_size[1]])

    def update_position(self, acceleration, steering, dt=1./60.):
        # Set acceleration and steering
        self.acceleration = max(min(acceleration, self.conf.acceleration_range[1]), self.conf.acceleration_range[0])
        self.angle_speed = max(min(steering, self.conf.steering_angle_range[1]), self.conf.steering_angle_range[0])

        # Update position and angle (Explicit Euler)
        old_speed = self.speed
        self.speed += self.acceleration * dt
        self.speed *= (1 - self.friction * dt)
        self.speed = max(min(self.speed, self.conf.velocity_range[1]), self.conf.velocity_range[0])
        # The speed can't change sign
        if self.speed * old_speed < 0:
            self.speed = 0
        if abs(self.speed) < 0.1: self.speed = 0.
        self.angle += self.angle_speed * dt
        angle = self.angle * np.pi / 180.
        self.position += self.speed * dt * np.array([np.cos(angle), np.sin(angle)])

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

    def get_front_position(self):
        angle = self.angle * np.pi / 180.
        return self.position + self.size[0] * 0.5 * np.array([np.cos(angle), np.sin(angle)])

    
class SweeperEnv(gym.Env):

    ACTION_INDEX_ACCELERATION = 0
    ACTION_INDEX_STEERING = 1

    def __init__(self, sweeper_config: SweeperConfig, reward_config: RewardConfig, render_options: RenderOptions, resolution: float = 1, debug: bool = False):
        self.debug = debug
        self.sweeper_config = sweeper_config
        self.render_options = render_options
        self.reward_config = reward_config
        self.resolution = resolution

        # Init everything
        sweeper_config.scale(resolution)
        self.init_gym(sweeper_config)
        self.init_map_and_sweeper(sweeper_config, resolution)
        if self.render_options.render:
            self.init_pygame()

        # Reset the environment
        self.reset()

    def init_gym(self, sweeper_config: SweeperConfig) -> None:
        # Extract from config
        accel_low, accel_high = sweeper_config.acceleration_range
        steer_low, steer_high = sweeper_config.steering_angle_range
        
        # Action Space:
        if sweeper_config.action_type == "continuous":
            # - Acceleration (continuous) # m/s^2
            # - Steering (continuous)     # degrees/s
            self.action_space = gym.spaces.Box(
                low=np.array([accel_low, steer_low]), high=np.array([accel_high, steer_high])
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
        else:
            raise Exception("Unknown action type: " + sweeper_config.action_type)
        self.action_to_acceleration_and_steering = self.get_action_to_acceleration_and_steering_fn()
        
        # State space:
        # - Position (x, y) (continuous)
        # - Velocity (x, y) (continuous)
        # - Angle (continuous)
        # - Steering (continuous)
        # - Grid (discrete 2D n*n)
        #    - 0: Empty
        #    - 1: Obstacle
        #    - 2: Cleaned
        # - Distances to closest obstacle (M directions) [Contraint -> Max v distance: max_distance]
        self.observation_space = None # TODO


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

        # Create sweeper
        self.sweeper = Sweeper(sweeper_config)

    def _get_observation(self):
        if self.sweeper_config.observation_type == 'dict':
            # Get the observation (copy position, speed, acceleration, angle, steering)
            # Return as a dict
            return {
                "position": self.sweeper.position.copy(),
                "speed": self.sweeper.speed,
                "acceleration": self.sweeper.acceleration,
                "angle": self.sweeper.angle,
                "steering": self.sweeper.angle_speed,
                "grid": self.map.grid,
                "distances": self.radars.copy()
            }
        elif self.sweeper_config.observation_type == 'torch-no-grid':
            # Get the observation (copy position, speed, acceleration, angle, steering)
            # Return as a torch tensor
            return [
                self.sweeper.speed / self.sweeper_config.velocity_range[1],
                self.sweeper.acceleration / self.sweeper_config.acceleration_range[1],
                np.cos(np.deg2rad(self.sweeper.angle)),
                np.sin(np.deg2rad(self.sweeper.angle)),
                *(self.radars / self.sweeper_config.radar_max_distance)
            ]
        elif self.sweeper_config.observation_type == 'torch-grid':
            raise NotImplementedError

    def get_action_to_acceleration_and_steering_fn(self):
        if self.sweeper_config.action_type == "continuous":
            return lambda action: action
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


    def step(self, action, dt=1./60):
        """Returns (observation, reward, terminated, info)"""
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
            self.sweeper.position = prev_position + fmin * (collision_position - prev_position)
            self.sweeper.angle = prev_angle + fmin * (self.sweeper.angle - prev_angle)
            self.sweeper.speed = 0.
            self.sweeper.acceleration = 0.

        # Compute reward
        # Position of front of sweeper
        front_position = self.sweeper.get_front_position()
        new_area = self.map.apply_cleaning(front_position, width=1.6*self.sweeper.size[1], resolution=self.resolution)
        self.sweeper_positions.append([self.sweeper.position[0], self.sweeper.position[1]])
        reward = self.reward_config.factor_area_cleaned * new_area + self.reward_config.reward_per_second * dt
        if had_collision:
            reward += self.reward_config.reward_collision
        if self.sweeper.speed < 0:
            reward += self.reward_config.reward_backwards

        # Radars
        self.compute_radars()

        # Update stats
        self.stats.update(new_area=new_area, new_reward=reward, had_collision=had_collision, dt=dt)

        # Return observation, reward, terminated, truncated, info
        done = had_collision and self.reward_config.done_on_collision or self.stats.percentage_cleaned >= 90.0 or self.iter >= self.sweeper_config.num_max_steps
        return self._get_observation(), reward, done, False, {"collision": had_collision}

    def compute_radars(self):
        for i in range(len(self.radars)):
            radar_angle = self.sweeper.angle + 360. / len(self.radars) * i
            self.radars[i] = self.map.compute_distance_to_closest_obstacle(self.sweeper.position, radar_angle, int(self.sweeper_config.radar_max_distance * self.resolution)) / self.resolution

    def check_collision(self):
        return self.map.check_collision(self.sweeper.get_bounding_box())

    def reset(self, new_map=True, generate_new_map=False, *args):
        """Reset the environment and return an initial observation and info."""
        self.iter = 0
        self.stats = SweeperStats()

        # Reset map
        self.render_options.first_render = True
        self.init_map_and_sweeper(sweeper_config=self.sweeper_config, resolution=self.resolution, new_map=new_map, generate_new_map=generate_new_map)
        self.map.cleaning_path = []
        self.stats.area_empty = self.map.get_empty_area(resolution=self.resolution)

        # Radars
        self.radars = np.zeros(self.sweeper_config.num_radars)

        # Reset sweeper by setting it's position to a random empty cell
        empty_cells = self.map.get_empty_tiles()
        collision = True
        while collision:
            self.sweeper.position = empty_cells[np.random.randint(len(empty_cells))].astype(float)
            collision = self.check_collision()
            self.compute_radars()
            if len(np.where(self.radars < self.sweeper_config.spawn_min_distance_to_wall)[0]) > 0:
                collision = True
            
        self.sweeper.angle = np.random.randint(360)
        self.sweeper.speed = 0
        self.sweeper.acceleration = 0
        self.sweeper.angle_speed = 0
        self.sweeper_positions = []

        # Return observation
        return self._get_observation(), {"collision": False,}



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
        # Render map
        if self.render_options.show_path or self.render_options.show_distance_sensors: self.render_options.first_render = True
        self.map.display(self.sweeper, self.screen, rerender=self.render_options.first_render)
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
            
        # Dislay distance sensors
        if self.render_options.show_distance_sensors:
            for i in range(self.sweeper_config.num_radars):
                distance = self.radars[i]
                # Draw the radar if the distance < max_distance
                if distance < self.sweeper_config.radar_max_distance:
                    angle = self.sweeper.angle + 360 / self.sweeper_config.num_radars * i
                    direction = self.resolution * distance * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                    pygame.draw.line(self.screen, self.render_options.distance_sensors_color, self.render_options.cell_size * sweeper_pos, self.render_options.cell_size * (sweeper_pos + direction), width=2) 
                    pygame.draw.circle(self.screen, self.render_options.distance_sensors_color, self.render_options.cell_size * (sweeper_pos + direction), 5)
        # Updates the screen
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
        if clock is not None:
            self.display_value('FPS', clock.get_fps(), int(self.render_options.width * 0.74), 70, color=(0, 255, 0))

        # Display time, area cleaned, percentage of the map cleaned and number of collisions in the middle in dark red
        self.display_value('Time', self.stats.time, int(self.render_options.width * 0.37), 10, color=(255, 0, 0))
        self.display_value('Area / seconds', self.stats.area_cleaned / self.stats.time, int(self.render_options.width * 0.37), 30, color=(255, 0, 0))
        self.display_value('Perc. cleaned', self.stats.percentage_cleaned, int(self.render_options.width * 0.37), 50, color=(255, 0, 0))
        self.display_value('Num. collisions', self.stats.collisions, int(self.render_options.width * 0.37), 70, color=(255, 0, 0))





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
        # - W: toggles the display of the distance sensors
        # - N: creates a new map
        # - +: increases the speed of the simulation
        # - -: decreases the speed of the simulation

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                sys.exit()
            if event.key == pygame.K_r:
                observation = self.reset(new_map=False)
            if event.key == pygame.K_t:
                observation = self.reset(new_map=True)
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
                print("W: Toggle distance sensors")
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
                self.render_options.show_distance_sensors = not self.render_options.show_distance_sensors
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

    # Implements a random agent for our gym environment
    sweeper_config = SweeperConfig()
    env = SweeperEnv(sweeper_config=sweeper_config, reward_config=RewardConfig(), render_options=RenderOptions(), resolution = 2.0, debug=False)
    observation, _ = env.reset()
    rewards = []
    cum_rewards = []
    cum_reward = 0
    positions_list = []
    past_action = (0, 0)

    dt = 1./60
    clock = pygame.time.Clock()

    while True:
        steering = 0
        acceleration = 0

        # Edit map
        left, middle, right = pygame.mouse.get_pressed()
        if left:
            # Get the position of the mouse
            mouse_pos = pygame.mouse.get_pos()
            # Get the cell position
            cell_pos = (mouse_pos[0] // env.render_options.cell_size, mouse_pos[1] // env.render_options.cell_size)
            env.map.set_cell_value(cell_pos[0], cell_pos[1], 1)
        if right:
            # Get the position of the mouse
            mouse_pos = pygame.mouse.get_pos()
            # Get the cell position
            cell_pos = (mouse_pos[0] // env.render_options.cell_size, mouse_pos[1] // env.render_options.cell_size)
            env.map.set_cell_value(cell_pos[0], cell_pos[1], 0)


        # Checks for event to close the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            else:
                env.process_pygame_event(event)

        # Checks for key pressed
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            steering = sweeper_config.steering_angle_range[0]
        if keys[pygame.K_RIGHT]:
            steering = sweeper_config.steering_angle_range[1]
        if keys[pygame.K_UP]:
            acceleration = sweeper_config.acceleration_range[1]
        if keys[pygame.K_DOWN]:
            acceleration = sweeper_config.acceleration_range[0]


        clock.tick(60)
        action = (acceleration, steering)
        # TODO: Use a random action. This is just for the baseline model
        # action = env.action_space.sample()
        # FILTER = 1 - 0.03 * ACCEL_BY
        # action = (FILTER * past_action[0] + (1 - FILTER) * action[0], FILTER * past_action[1] + (1 - FILTER) * action[1])
        # past_action = action
        observation, reward, terminated, truncated, info = env.step(action, dt=dt)
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
