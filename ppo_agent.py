import gym
from stable_baselines3 import PPO
from env import SweeperEnv
from config import SweeperConfig, RewardConfig, RenderOptions
import argparse
import numpy as np
from tqdm import tqdm
parser = argparse.ArgumentParser()


# parser.add_argument("--num_episodes", type=int, default=500)
# parser.add_argument("--log_every", type=int, default=0)
# parser.add_argument("--plot", action="store_true")
# parser.add_argument("--reset_map_every", type=int, default=1)
# parser.add_argument("--model_name", type=str, default="model")
# args = parser.parse_args()
# num_episodes = args.num_episodes
# log_every = args.log_every


# Create the environment
sweeper_config = SweeperConfig(observation_type='torch-grid', action_type='discrete-minimum', num_max_steps=5000, num_radars=20)
reward_config = RewardConfig(done_on_collision=True, reward_per_second=0, reward_per_step=-0.1, reward_backwards=-3, reward_collision=-1000, factor_area_cleaned=10.0)
render_options = RenderOptions(render=True)
env = SweeperEnv(sweeper_config=sweeper_config, reward_config=reward_config, render_options=render_options, resolution = 2.0, debug=False)


model = PPO("MultiInputPolicy", env, verbose=1)

# Prints the model architecture
print(model.policy)

for _ in range(10):
    model.learn(total_timesteps=50_000)

    vec_env = model.get_env()

    #env.set_render_options(RenderOptions(render=True, width=800))


    obs = vec_env.reset()
    done = False
    for i in tqdm(range(10000), desc="Testing"):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        if done:
            break
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()

    # Save the agent
    model.save("models/ppo_sweeper_i")

env.close()
