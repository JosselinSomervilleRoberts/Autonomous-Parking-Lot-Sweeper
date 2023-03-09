import sys
from stable_baselines3 import PPO, DQN, DDPG, TD3, SAC, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from env import SweeperEnv
from config import SweeperConfig, RewardConfig, RenderOptions
import argparse
import numpy as np
from tqdm import tqdm
import os
import pygame
from user_utils import print_with_color, ask_yes_no_question, warn, error
import torch.nn as nn
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

parser = argparse.ArgumentParser()

# Logging
parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard logging")
parser.add_argument("--tensorboard-log", type=str, default="./ppo_tensorboard/", help="Tensorboard log dir")
parser.add_argument("--verbose", type=int, default=1, help="Verbose mode (0: no output, 1: INFO)")
parser.add_argument("--save_freq", type=int, default=50000, help="Save model every x steps (0: no checkpoint)")
parser.add_argument("--save_path", type=str, default="./models/", help="Path to save the model")
parser.add_argument("--load_path", type=str, default=None, help="Path to load the model")

# Environment
parser.add_argument("--observation_type", type=str, default="simple-radar-cnn", help="Observation type", choices=["simple", "simple-double-radar", "simple-radar-cnn", "grid-only", "complex"])
parser.add_argument("--action_type", type=str, default="discrete-5", help="Action type")
parser.add_argument("--env_max_steps", type=int, default=2048, help="Max steps per episode")

# Reward
parser.add_argument("--reward_collision", type=float, default=-1024, help="Reward for collision")
parser.add_argument("--reward_per_step", type=float, default=-0.1, help="Reward per step")
parser.add_argument("--reward_per_second", type=float, default=0, help="Reward per second")
parser.add_argument("--reward_area_total", type=float, default=4096, help="Reward factor for area")
parser.add_argument("--reward_backwards", type=float, default=-0.5, help="Reward for going backwards")
parser.add_argument("--reward_idle", type=float, default=-0.2, help="Reward for idling")
parser.add_argument("--not_done_on_collision", action="store_true", help="Not done on collision")

# Algorithm
parser.add_argument("--algorithm", type=str, default="PPO", help="RL Algorithm", choices=["PPO", "DQN", "DDPG", "TD3", "SAC", "A2C"])
parser.add_argument("--policy", type=str, default="MultiInputPolicy", help="Policy type", choices=["MlpPolicy", "CnnPolicy", "MultiInputPolicy"])
parser.add_argument("--total_timesteps", type=int, default=5_120_000, help="Number of timesteps")
parser.add_argument("--n_iter_learn", type=int, default=1, help="Number of times to run the learning process")
parser.add_argument("--disable_visual_test", action="store_true", help="Disable visual test")

# Algorithm specific
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps in each rollout")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator")
parser.add_argument("--ent_coef", type=float, default=0.0, help="Entropy coefficient for the loss calculation")
parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient for the loss calculation")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="The maximum value for the gradient clipping")

args = parser.parse_args()

# Check validity of action_type
if args.action_type == "discrete" and args.algorithm != "DQN":
    error("Discrete action type is only supported by DQN algorithm")
    sys.exit(1)
elif args.action_type == "continuous" and args.algorithm == "DQN":
    error("Continuous action type is not supported by DQN algorithm")
    sys.exit(1)
elif "multi-discrete" in args.action_type and args.algorithm != "DDPG":
    error("Multi-discrete action type is only supported by DDPG algorithm")
    sys.exit(1)
elif args.action_type not in ["discrete", "continuous", "multi-discrete", "multi-discrete-continuous"] \
    and "multi-discrete" not in args.action_type and "discrete" not in args.action_type:
    error("Invalid action type")
    sys.exit(1)

# Check validity of observation_type
if args.observation_type == "simple" and args.policy != "MlpPolicy":
    error("Simple observation type is only supported by MlpPolicy")
    sys.exit(1)
elif args.observation_type == "grid-only" and args.policy != "CnnPolicy":
    error("Grid-only observation type is only supported by CnnPolicy")
    sys.exit(1)
elif args.observation_type == "complex" and args.policy != "MultiInputPolicy":
    error("Complex observation type is only supported by MultiInputPolicy")
    sys.exit(1)

# Print the arguments
print_with_color(f"""\n==================== Arguments ====================
tensorboard: {args.tensorboard}
tensorboard_log: {args.tensorboard_log}
verbose: {args.verbose}

observation_type: {args.observation_type}
action_type: {args.action_type}
env_max_steps: {args.env_max_steps}

reward_collision: {args.reward_collision}
reward_per_step: {args.reward_per_step}
reward_per_second: {args.reward_per_second}
reward_area_total: {args.reward_area_total}
reward_backwards: {args.reward_backwards}
reward_idle: {args.reward_idle}
done_on_collision: {not args.not_done_on_collision}

algorithm: {args.algorithm}
policy: {args.policy}
total_timesteps: {args.total_timesteps}
n_iter_learn: {args.n_iter_learn}

learning_rate: {args.learning_rate}
n_steps: {args.n_steps}
batch_size: {args.batch_size}
gamma: {args.gamma}
gae_lambda: {args.gae_lambda}
ent_coef: {args.ent_coef}
vf_coef: {args.vf_coef}
max_grad_norm: {args.max_grad_norm}
===================================================\n""", color='purple')

# args algorithm to model_type
model_type = None
if args.algorithm == "PPO":
    model_type = PPO
elif args.algorithm == "DQN":
    model_type = DQN
elif args.algorithm == "DDPG":
    model_type = DDPG
elif args.algorithm == "TD3":
    model_type = TD3
elif args.algorithm == "SAC":
    model_type = SAC
elif args.algorithm == "A2C":
    model_type = A2C


# Create the environment
sweeper_config = SweeperConfig(observation_type=args.observation_type, action_type=args.action_type, num_max_steps=args.env_max_steps)
reward_config = RewardConfig(done_on_collision=(not args.not_done_on_collision), reward_collision=args.reward_collision, reward_per_step=args.reward_per_step, reward_per_second=args.reward_per_second, reward_area_total=args.reward_area_total, reward_backwards=args.reward_backwards, reward_idle=args.reward_idle)
render_options = RenderOptions(render=True)
print_with_color(str(sweeper_config) + "\n", color='blue')
print_with_color(str(reward_config) + "\n", color='yellow')
print_with_color(str(render_options) + "\n", color='darkcyan')

env = SweeperEnv(sweeper_config=sweeper_config, reward_config=reward_config, render_options=render_options, resolution = 2.0, debug=False)
check_env(env, warn=True)

if args.tensorboard and os.name == 'posix': # Checks if os is Linux
    # Asks if the user wants to delete the tensorboard log
    print_with_color("\nTensorboard log directory: " + args.tensorboard_log, color='green')
    delete = ask_yes_no_question("Do you want to delete the tensorboard log directory?", ["bold", "green"])
    if delete:
        print_with_color("Deleting tensorboard log directory...", color='green')
        os.system("rm -rf " + args.tensorboard_log)
        print_with_color("Deleted tensorboard log directory.\n", color='green')
    else:
        print_with_color("Not deleting tensorboard log directory.\n", color='green')


class Net(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super(Net, self).__init__(observation_space, features_dim)
        
        # RADARS
        # get shape of observation_space["radars"]
        (self.n_cell, self.n_dir, self.n_r) = observation_space["radars"].shape
        self.channel_inter_size = 16
        # Step 1: per direction, convolve type of cells and different radius: (n_cell, n_dir, n_r) -> (channel_inter_size, n_dir, 1)
        self.conv1 = nn.Conv2d(in_channels=self.n_cell, out_channels=self.channel_inter_size, kernel_size=(1, self.n_r), padding=0, bias=True, stride=1)
        print("conv1 weight shape: ", self.conv1.weight.shape)

        # Step 2: convolve directions: (channel_inter_size, n_dir, 1) -> (channel_inter_size_2, n_dir_2, 1)
        self.n_dir_2 = 12
        self.kernel_size_2 = 7
        self.channel_inter_size_2 = 8
        self.stride2 = int(self.n_dir / self.n_dir_2)
        self.padding2 = int((self.stride2 * self.n_dir_2 - 1 + self.kernel_size_2 - self.n_dir) / 2)
        self.conv2 = nn.Conv2d(in_channels=self.channel_inter_size, out_channels=self.channel_inter_size_2, kernel_size=(self.kernel_size_2, 1), padding_mode="circular", padding=(self.padding2,0), bias=True, stride=(self.stride2, 1))
        self.first_run = True

        # Step 3: Reshape: (channel_inter_size_2, n_dir_2, 1) -> (n_dir_2 * dir_inter_size)


        # OTHER
        # get shape of observation_space["other"]
        self.shape_other = observation_space["other"].shape[0]

        # Concatenate
        # self.fc1 = nn.Linear(in_features=dir_inter_size + shape_other, out_features=features_dim)
        self.fc1 = nn.Linear(in_features=self.n_dir_2*self.channel_inter_size_2 + self.shape_other, out_features=features_dim)
        #self.features_dim = features_dim

    def forward(self, x):
        if self.first_run:
            # Conv 1
            weight1 = np.zeros((self.channel_inter_size, self.n_cell, 1, self.n_r), dtype=np.float32)
            for i in range(6):
                weight1[i, 0, 0, i] = 1.
                weight1[6+i, 2, 0, i] = 1.
            self.conv1.weight = torch.nn.Parameter(torch.FloatTensor(weight1), requires_grad=False)
            self.conv1.bias = torch.nn.Parameter(torch.FloatTensor([0] * 16), requires_grad=False)
            for param in self.conv1.parameters():
                param.requires_grad = False

            # Conv 2
            weight2 = np.zeros((8, 16, 7, 1))
            weight2[0,0,3,:] = 1.
            weight2[1,0,2:5,:] = 1./3.
            weight2[2,0:2,1:6,:] = 1./10.
            weight2[3,0:6,0:7,:] = 1./42.
            weight2[0,6,3,:] = 1.
            weight2[1,6,2:5,:] = 1./3.
            weight2[2,6:8,1:6,:] = 1./10.
            weight2[3,6:12,0:7,:] = 1./42.
            self.conv2.weight = torch.nn.Parameter(torch.FloatTensor(weight2), requires_grad=False)
            self.conv2.bias = torch.nn.Parameter(torch.FloatTensor([0] * 8), requires_grad=False)
            for param in self.conv2.parameters():
                param.requires_grad = False

            # FC 1
            weight3 = np.zeros(self.fc1.weight.shape, dtype=np.float32)
            for i in range(self.n_dir_2*self.channel_inter_size_2 + self.shape_other):
                weight3[i, i] = 1.
            self.fc1.weight = torch.nn.Parameter(torch.FloatTensor(weight3), requires_grad=False)
            self.fc1.bias = torch.nn.Parameter(torch.FloatTensor([0] * self.features_dim), requires_grad=False)
            for param in self.fc1.parameters():
                param.requires_grad = False

            # Dont run this again
            self.first_run = False

        radar = x["radars"]
        other = x["other"]
        batch_size = radar.shape[0]

        # RADARS
        # Step 1: per direction, convolve type of cells and different radius: (n_cell, n_dir, n_r) -> (channel_inter_size, n_dir, 1)
        # print("1. radar.shape: ", radar.shape)
        radar = F.relu(self.conv1(radar))
        # Step 2: Reshape: (channel_inter_size, n_dir, 1) -> (1, n_dir, channel_inter_size)
        # print("2. radar.shape: ", radar.shape)
        #radar = radar.swapaxes(-1, -3)
        # print("3. radar.shape: ", radar.shape)
        # Step 3: convolve directions: (1, n_dir, channel_inter_size) -> (channel_inter_size_2, dir_inter_size, 1)
        radar = F.relu(self.conv2(radar))
        # print("4. radar.shape: ", radar.shape)
        # Step 4: Reshape: (channel_inter_size_2, dir_inter_size, 1) -> (1, dir_inter_size, channel_inter_size_2)
        #radar = radar.swapaxes(-1, -3)
        # print("5. radar.shape: ", radar.shape)
        # Step 5: convolve directions: (1, dir_inter_size, channel_inter_size_2) -> (1, dir_inter_size, 1)
        #radar = F.relu(self.conv3(radar))
        # print("6. radar.shape: ", radar.shape)
        # Step 6: Reshape: (1, dir_inter_size, 1) -> (dir_inter_size)
        radar = radar.reshape(batch_size, -1)
        # print("7. radar.shape: ", radar.shape)

        # OTHER
        # print("1. other.shape: ", other.shape)
        other = other.reshape(batch_size, -1)
        # print("2. other.shape: ", other.shape)

        # Concatenate
        c = torch.concat((radar, other), dim=-1)
        # print("1. c.shape: ", c.shape)
        c = F.tanh(self.fc1(c))
        # print("2. c.shape: ", c.shape)

        return c

policy_kwargs = {
    'activation_fn': nn.Tanh,
    'net_arch':[64, dict(pi=[64, 32], vf=[64, 32])],
    "features_extractor_kwargs": dict(features_dim=128),
    'features_extractor_class':Net,
}


# Create the model
checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=args.save_path)
tensorboard_log = args.tensorboard_log if args.tensorboard else None
model = model_type(policy=args.policy,
    env=env,
    # learning_rate=args.learning_rate,
    #n_steps=args.n_steps,
    batch_size=args.batch_size,
    # n_epochs=args.n_epochs,
    gamma=args.gamma,
    # gae_lambda=args.gae_lambda,
    # clip_range=args.clip_range,
    # clip_range_vf=args.clip_range_vf,
    # normalize_advantage=args.normalize_advantage,
    # ent_coef=args.ent_coef,
    # vf_coef=args.vf_coef,
    # max_grad_norm=args.max_grad_norm,
    # use_sde=args.use_sde,
    # sde_sample_freq=args.sde_sample_freq,
    # target_kl=args.target_kl,
    tensorboard_log=tensorboard_log,
    policy_kwargs=policy_kwargs,
    verbose=args.verbose,
    # seed=args.seed,
    # device=args.device,
    # _init_setup_model=True
)

if args.tensorboard: # Checks if tensorboard logging is enabled
    print_with_color("\nTensorboard log directory: " + args.tensorboard_log, color='green')
    print_with_color("You can try launching tensorboard manually with the following command:", color='green')
    print_with_color("tensorboard --logdir=./ppo_tensorboard/ --port=6006", color=['bold', 'green'])
    print_with_color("You can view the logs at http://localhost:6006/", color='green')
    print_with_color("Press Ctrl+C to stop tensorboard.", color='green')

# Prints the model architecture
print_with_color("\nModel architecture:", color=['bold', 'purple'])
print_with_color(str(model.policy), color='purple')
print("")

# Creates a vectorized environment from env with n_envs copies of the environment
# env = DummyVecEnv([lambda: SweeperEnv(sweeper_config=sweeper_config, reward_config=reward_config, render_options=render_options, resolution = 2.0, debug=False) for _ in range(4)])



for j in range(args.n_iter_learn):
    # Train the agent
    if args.load_path is None:
        print_with_color("Training model " + str(j) + " of " + str(args.n_iter_learn) + "...", color='green')
        model.learn(total_timesteps=args.total_timesteps, callback=[checkpoint_callback])
        # Save the agent
        model.save(args.save_path + args.algorithm + "_model_" + str(j))
    else:
        print("Loading model from " + args.load_path)
        #help(model.load)
        model.set_parameters(args.load_path) 

    # Test the agent
    vec_env = model.get_env()
    # TODO: Do something for the rendering

    obs = vec_env.reset()

    if not args.disable_visual_test:
        clock = pygame.time.Clock()
        while True:
            if pygame.get_init():
                # Checks for event to close the window
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    else:
                        env.process_pygame_event(event)
            clock.tick(sweeper_config.action_frequency)

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render()

            # VecEnv resets automatically
            if done:
                obs = vec_env.reset()
    else:
        print("Iter " + str(j) + "done")
        print("To see the visual test, run the script with the --disable-visual-test flag disabled.")


env.close()
