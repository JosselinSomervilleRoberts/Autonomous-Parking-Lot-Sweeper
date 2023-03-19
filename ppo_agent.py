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


parser = argparse.ArgumentParser()

# Logging
parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard logging")
parser.add_argument("--tensorboard-log", type=str, default="./ppo_tensorboard/", help="Tensorboard log dir")
parser.add_argument("--verbose", type=int, default=1, help="Verbose mode (0: no output, 1: INFO)")
parser.add_argument("--save_freq", type=int, default=50000, help="Save model every x steps (0: no checkpoint)")
parser.add_argument("--save_path", type=str, default="./models/", help="Path to save the model")
parser.add_argument("--load_path", type=str, default=None, help="Path to load the model")

# Environment
parser.add_argument("--observation_type", type=str, default="complex", help="Observation type", choices=["simple", "simple-double-radar", "grid-only", "complex"])
parser.add_argument("--action_type", type=str, default="continuous", help="Action type")
parser.add_argument("--env_max_steps", type=int, default=4096, help="Max steps per episode")
parser.add_argument("--env_num_radars", type=int, default=16, help="Number of radars")

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
env_num_radars: {args.env_num_radars}

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
sweeper_config = SweeperConfig(observation_type=args.observation_type, action_type=args.action_type, num_max_steps=args.env_max_steps, num_radars=args.env_num_radars)
reward_config = RewardConfig(done_on_collision=(not args.not_done_on_collision), reward_collision=args.reward_collision, reward_per_step=args.reward_per_step, reward_per_second=args.reward_per_second, reward_area_total=args.reward_area_total, reward_backwards=args.reward_backwards, reward_idle=args.reward_idle)
render_options = RenderOptions(render=True)
print_with_color(str(sweeper_config) + "\n", color='blue')
print_with_color(str(reward_config) + "\n", color='yellow')
print_with_color(str(render_options) + "\n", color='darkcyan')

env = SweeperEnv(sweeper_config=sweeper_config, reward_config=reward_config, render_options=render_options, resolution = 2.0, debug=False)

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

# Create the model
checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=args.save_path)
tensorboard_log = args.tensorboard_log if args.tensorboard else None
model = model_type(policy=args.policy,
    env=env,
    # learning_rate=args.learning_rate,
    #n_steps=args.n_steps,
    # batch_size=args.batch_size,
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
    # policy_kwargs={"net_arch": [128,256,256,128]},
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
