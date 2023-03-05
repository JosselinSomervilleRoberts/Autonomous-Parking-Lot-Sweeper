import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from metrics import get_patch_of_line
from game import SweeperGame
import time
from PIL import Image
from math import ceil

@dataclass
class SweeperConfig:
    acceleration_range      : Tuple[float, float]
    velocity_range          : Tuple[float, float]
    steering_delta_range    : float
    steering_angle_range    : Tuple[float, float]
    max_distance            : float
    friction                : float
    sweeper_size            : Tuple[int, int]


def between(minimum, maximum):
    def func(val):
        return minimum <= val <= maximum
    return func


class Sweeper:
    def __init__(self, config : SweeperConfig):
        # Linear
        self.position = np.array([0., 0.])
        self.speed = 0.
        self.acceleration = 0.
        self.friction = config.friction

        # Rotation
        self.angle = 0.
        self.angle_speed = 0.

        # Vehicle size
        self.size = config.sweeper_size

    def update_position(self, acceleration, steering, dt=1./60.):
        # Set acceleration and steering
        self.acceleration = acceleration
        self.angle_speed = steering

        # Update position and angle (Explicit Euler)
        self.speed += self.acceleration * dt
        self.speed *= (1 - self.friction * dt)
        self.angle += self.angle_speed * dt
        angle = self.angle * np.pi / 180.
        self.position += self.speed * dt * np.array([np.cos(angle), np.sin(angle)])

    def get_bounding_box(self):
        # Get bounding box of the sweeper (the position is the center of the sweeper)
        x, y = self.position
        w, h = self.size
        angle = self.angle * np.pi / 180.
        return np.array([
            [x + w/2 * np.cos(angle) - h/2 * np.sin(angle), y + w/2 * np.sin(angle) + h/2 * np.cos(angle)],
            [x - w/2 * np.cos(angle) - h/2 * np.sin(angle), y - w/2 * np.sin(angle) + h/2 * np.cos(angle)],
            [x - w/2 * np.cos(angle) + h/2 * np.sin(angle), y - w/2 * np.sin(angle) - h/2 * np.cos(angle)],
            [x + w/2 * np.cos(angle) + h/2 * np.sin(angle), y + w/2 * np.sin(angle) - h/2 * np.cos(angle)]
        ])

    
class SweeperEnv(gym.Env):

    ACTION_INDEX_ACCELERATION = 0
    ACTION_INDEX_STEERING = 1

    def __init__(self, config : SweeperConfig):
        # Extract from config
        accel_low, accel_high = config.acceleration_range
        steer_low, steer_high = config.steering_angle_range
        max_distance = config.max_distance
        
        # Action Space:
        # - Acceleration (continuous) # m/s^2
        # - Steering (continuous)     # degrees/s
        self.action_space = gym.spaces.Box(
            low=np.array([accel_low, steer_low]), high=np.array([accel_high, steer_high])
        )
        self.observation_space = gym.spaces.Discrete(2)
        self.max_distance = max_distance
        
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

        # Reward parameters
        self.curr_covered_area = 0
        self.reward_iter_penalty = 0.01

        # Create sweeper agent
        self.sweeper = Sweeper(config)
        self.sweeper_positions = [[self.sweeper.position[0], self.sweeper.position[1]]]

        # Game
        self.game = SweeperGame(width=128, height=128, sweeper=self.sweeper, cell_size=6)

        # Reset
        self.reset()

    def _get_observation(self):
        return [self.sweeper.position[0], self.sweeper.position[1]]

    def step(self, action: Tuple[float, float], dt=1./60):
        # Returns (observation, reward, terminated, truncated, info)
        # Update sweeper
        self.iter += 1
        acceleration = action[self.ACTION_INDEX_ACCELERATION]
        steering = action[self.ACTION_INDEX_STEERING]
        prev_position = self.sweeper.position.copy()
        prev_angle = self.sweeper.angle

        self.sweeper.update_position(acceleration, steering, dt)

        # Check for collision
        # If there is one, find with a binary search the last position where there was no collision
        if self.check_collision():
            collision_position = self.sweeper.position.copy()
            print("Collision at", collision_position, "from", prev_position, "with action", action)
            # Binary search between prev_position and collision_position
            N_BINARY_SEARCH = 10
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
            return self._get_observation(), -10, False, False, {"collision": True}

        # Compute reward
        self.sweeper_positions.append([self.sweeper.position[0], self.sweeper.position[1]])
        self.patch = get_patch_of_line(self.sweeper_positions, width=0.7*self.sweeper.size[1])
        area = self.patch.area
        reward = area - self.curr_covered_area - self.reward_iter_penalty
        self.curr_covered_area = area
        return self._get_observation(), reward, False, False, {"collision": False}

    def check_collision(self):
        return self.game.map.check_collision(self.sweeper.get_bounding_box())

    def reset(self, *args):
        # Returns observation
        self.iter = 0
        self.game.map.init_random()

        # Reset sweeper by setting it's position to a random empty cell
        empty_cells = self.game.map.get_empty_tiles()
        collision = True
        while collision:
            self.sweeper.position = empty_cells[np.random.randint(len(empty_cells))].astype(float)
            collision = self.check_collision()
        self.sweeper.angle = np.random.randint(360)
        self.sweeper.speed = 0
        self.sweeper.acceleration = 0
        self.sweeper.angle_speed = 0

    def render(self):
        self.game.render()#self.sweeper.position, self.sweeper.angle, self.sweeper_positions, self.patch)


if __name__ == "__main__":
    import pygame
    import sys

    # Implements a random agent for our gym environment
    config = SweeperConfig(
        # acceleration_range=(-0.25, 0.25),
        acceleration_range=(-5*6*6,5*6*6),  # units/s**2
        velocity_range=(-1*6, 1*6),           # units/s
        steering_delta_range=50,
        steering_angle_range=(-90*3, 90*3),   # degrees/s
        max_distance=50,                        # units
        friction=0.1*6*6,                        # 1/s
        sweeper_size=np.array([6, 3])                   # cells
    )
    env = SweeperEnv(config=config)
    observation = env.reset()
    rewards = []
    cum_rewards = []
    cum_reward = 0
    positions_list = []
    past_action = (0, 0)

    ACCEL_BY = 1 # Makes the game 5 times faster
    dt = 1./60

    while True:
        steering = 0
        acceleration = 0

        # Checks for event to close the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # Checks for key pressed
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            steering = config.steering_angle_range[0]
        if keys[pygame.K_RIGHT]:
            steering = config.steering_angle_range[1]
        if keys[pygame.K_UP]:
            acceleration = config.acceleration_range[1]
        if keys[pygame.K_DOWN]:
            acceleration = config.acceleration_range[0]


        time.sleep(dt)
        action = (acceleration, steering)
        # TODO: Use a random action. This is just for the baseline model
        # action = env.action_space.sample()
        # FILTER = 1 - 0.03 * ACCEL_BY
        # action = (FILTER * past_action[0] + (1 - FILTER) * action[0], FILTER * past_action[1] + (1 - FILTER) * action[1])
        # past_action = action
        observation, reward, terminated, truncated, info = env.step(action, dt=dt*ACCEL_BY)
        env.render()
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
