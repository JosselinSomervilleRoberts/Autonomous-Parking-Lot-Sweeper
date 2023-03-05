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
        self.position = np.array([150., 500.])
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
        self.game = SweeperGame(width=100, height=100, sweeper=self.sweeper, cell_size=8)

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

        self.sweeper.update_position(acceleration, steering, dt)

        # Check for collision and prevent position update
        if self.check_collision(prev_position, self.sweeper.position):
            self.sweeper.position = prev_position
            return self._get_observation(), -10, False, False, {"collision": True}

        # Compute reward
        self.sweeper_positions.append([self.sweeper.position[0], self.sweeper.position[1]])
        self.patch = get_patch_of_line(self.sweeper_positions, width=0.7*self.sweeper.size[1])
        area = self.patch.area
        reward = area - self.curr_covered_area - self.reward_iter_penalty
        self.curr_covered_area = area
        return self._get_observation(), reward, False, False, {"collision": False}

    def check_collision(self, prev_position, curr_position):
        return False

    def reset(self, *args):
        # Returns observation
        self.iter = 0
        self.game.map.init_random()

    def render(self):
        self.game.render(self.sweeper.position, self.sweeper.angle, self.sweeper_positions, self.patch)


if __name__ == "__main__":
    # Implements a random agent for our gym environment
    config = SweeperConfig(
        # acceleration_range=(-0.25, 0.25),
        acceleration_range=(-5*6*6,5*6*6),  # units/s**2
        velocity_range=(-1*6, 1*6),           # units/s
        steering_delta_range=50,
        steering_angle_range=(-90*3, 90*3),   # degrees/s
        max_distance=50,                        # units
        friction=0.1*6,                        # 1/s
        sweeper_size=(50, 25)                   # units
    )
    env = SweeperEnv(config=config)
    observation = env.reset()
    rewards = []
    cum_rewards = []
    cum_reward = 0
    positions_list = []
    past_action = (0, 0)

    ACCEL_BY = 10 # Makes the game 5 times faster
    dt = 1./60

    for _ in range(500):
        time.sleep(dt)
        # action = (0.25, 1.0)    # env.action_space.sample()
        # TODO: Use a random action. This is just for the baseline model
        action = env.action_space.sample()
        FILTER = 1 - 0.03 * ACCEL_BY
        action = (FILTER * past_action[0] + (1 - FILTER) * action[0], FILTER * past_action[1] + (1 - FILTER) * action[1])
        past_action = action
        observation, reward, terminated, truncated, info = env.step(action, dt=dt*ACCEL_BY)
        env.render()
        if terminated:
            observation = env.reset()
        # Keep tracks of rewards
        rewards.append(reward)
        cum_reward += reward
        cum_rewards.append(cum_reward)
        positions_list.append(observation)

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
