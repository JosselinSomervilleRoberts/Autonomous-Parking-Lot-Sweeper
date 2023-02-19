import gymnasium as gym
import numpy as np
import pygame
from typing import Tuple, TypedDict
from dataclasses import dataclass
from metrics import compute_area_of_path
from game import SweeperGame
import time

@dataclass
class SweeperConfig:
    acceleration_range : Tuple[float, float]
    velocity_range : Tuple[float, float]
    steering_delta_range : float
    steering_angle_range : Tuple[float, float]
    max_distance : float
    friction : float



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

    def update(self, acceleration, steering):
        # Set acceleration and steering
        self.acceleration = acceleration
        self.angle_speed = steering

        # Update position and angle (Explicit Euler)
        self.speed += self.acceleration
        self.speed *= (1 - self.friction)
        self.angle += self.angle_speed
        angle = self.angle * np.pi / 180.
        self.position += self.speed * np.array([np.cos(angle), np.sin(angle)])

    
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

        # Create sweeper
        self.sweeper = Sweeper(config)
        self.sweeper_positions = [[self.sweeper.position[0], self.sweeper.position[1]]]

        # Reset
        self.reset()

        # Game
        self.game = SweeperGame()

    def _get_observation(self):
        return [self.sweeper.position[0], self.sweeper.position[1]]

    def step(self, action: Tuple[float, float]):
        # Returns (observation, reward, done, info)

        # Update sweeper
        self.iter += 1
        acceleration = action[self.ACTION_INDEX_ACCELERATION]
        steering = action[self.ACTION_INDEX_STEERING]
        self.sweeper.update(acceleration, steering)

        # Compute reward
        self.sweeper_positions.append([self.sweeper.position[0], self.sweeper.position[1]])
        area = compute_area_of_path(self.sweeper_positions)
        reward = area - self.curr_covered_area - self.reward_iter_penalty
        self.curr_covered_area = area
        
        return self._get_observation(), reward, False, {}

    def reset(self):
        # Returns observation
        self.iter = 0

    def render(self):
        self.game.render(self.sweeper.position, self.sweeper.angle, self.sweeper_positions)

    
    
    
if __name__ == "__main__":
    # Implements a random agent for our gym environment
    config = SweeperConfig(
        acceleration_range=(-0.25, 0.25),
        velocity_range=(-1, 1),
        steering_delta_range=1,
        steering_angle_range=(-90, 90),
        max_distance=10,
        friction=0.1
    )
    env = SweeperEnv(config)
    observation = env.reset()
    rewards = []
    cum_rewards = []
    cum_reward = 0
    positions_list = []
    past_action = (0, 0)

    for _ in range(1000):
        time.sleep(0.01)
        env.render()
        action = (0.25, 1.0)#env.action_space.sample()
        FILTER = 0.9
        #action = (FILTER * past_action[0] + (1 - FILTER) * action[0], FILTER * past_action[1] + (1 - FILTER) * action[1])
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
    
        # Keep tracks of rewards
        rewards.append(reward)
        cum_reward += reward
        cum_rewards.append(cum_reward)
        positions_list.append(observation)

    # Plot rewards
    import matplotlib.pyplot as plt
    plt.plot(rewards, label="Rewards")
    plt.plot(cum_rewards, label="Cumulative Rewards")
    plt.legend()
    plt.show()

    # Plot positions
    positions = np.array(positions_list)
    plt.plot(positions[:, 0], positions[:, 1], label="Positions")
    plt.legend()
    plt.show()
