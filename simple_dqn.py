import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env import SweeperEnv
from config import SweeperConfig, RewardConfig, RenderOptions
import argparse
import numpy as np
from tqdm import tqdm
parser = argparse.ArgumentParser()


parser.add_argument("--num_episodes", type=int, default=500)
parser.add_argument("--log_every", type=int, default=0)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--reset_map_every", type=int, default=1)
parser.add_argument("--model_name", type=str, default="model")
args = parser.parse_args()
num_episodes = args.num_episodes
log_every = args.log_every



# Create the environment
sweeper_config = SweeperConfig(observation_type='torch-no-grid', action_type='discrete-5', num_max_steps=800, num_radars=20)
reward_config = RewardConfig(done_on_collision=True, reward_per_second=0, reward_per_step=0.5, reward_backwards=-3, reward_collision=-1000, reward_area_total=10.0)
render_options = RenderOptions(render=False)
env = SweeperEnv(sweeper_config=sweeper_config, reward_config=reward_config, render_options=render_options, resolution = 2.0, debug=False)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)




Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return self.layer4(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.8
EPS_END = 0.05
TAU = 0.001
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

print("n_actions: ", n_actions)
print("n_observations: ", n_observations)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state, i_episode):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * i_episode / (num_episodes - 1)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def select_action_greedy(state):
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)


episode_rewards = []


def plot_rewards(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Rewards')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.ones(99) * reward_config.reward_collision, means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# Training loop
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

dt = 1./60

# Load model if exists
import os
name = args.model_name + ".pth"
if os.path.exists(name):
    print("Loading model...")
    policy_net.load_state_dict(torch.load(name))
else:
    for i_episode in tqdm(range(num_episodes), desc="Episode", disable=log_every != 0, smoothing=0):
        # Initialize the environment and get it's state
        new_map = args.reset_map_every > 0 and i_episode % args.reset_map_every == 0
        state, info = env.reset(new_map=new_map)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        cum_reward = 0
        for t in count():
            action = select_action(state, i_episode)
            observation, reward, terminated, truncated, _ = env.step(action.item(), dt=dt)
            cum_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_rewards.append(cum_reward)
                if args.plot: plot_rewards()
                break
        if log_every > 0 and i_episode % log_every == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, episode_rewards[-1], np.mean(episode_rewards[-log_every:])))

print('Complete')
plot_rewards(show_result=True)
plt.ioff()
plt.show()

# Save the model
torch.save(policy_net.state_dict(), name)

# Show the final result
# Set render to True to see the final result
env.render_options.render = True
env.init_pygame()
import pygame, sys
clock = pygame.time.Clock()

import time
last_time_sec = time.time()

while True:

    # Initialize the environment and get it's state
    state, info = env.reset(new_map=True)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():

        # Break if the user closes the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            env.process_pygame_event(event)

        # Frame rate
        clock.tick(60)

        action = select_action_greedy(state)
        cur_time_sec = time.time()
        dt = cur_time_sec - last_time_sec
        last_time_sec = cur_time_sec
        observation, reward, terminated, truncated, _ = env.step(action.item(), dt=dt)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Move to the next state
        state = next_state

        # Render the environment
        env.render()

        if done:
            break