import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make("LunarLander-v2")

# Create PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=100000, save_path=log_dir, name_prefix="rl_model_v2"
)
# Train the agent
model.learn(
    total_timesteps=500000,
    callback=[
        checkpoint_callback],)
