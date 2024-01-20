import os
import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


MODEL_PATH = "tmp/rl_model_v2_100000_steps.zip"

# Create and wrap the environment
env = gym.make("LunarLander-v2", render_mode="human")
env = DummyVecEnv([lambda: env])
#env = VecTransposeImage(env)

# Load the trained agent
model = PPO.load(MODEL_PATH, env=env)


# Evaluate the agent
for i in range(10):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        env.render("yes")
    print("Episode reward", episode_reward)
    
env.close()

