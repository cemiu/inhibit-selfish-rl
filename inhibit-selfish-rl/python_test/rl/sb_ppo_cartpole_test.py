import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from util.state_util import State

log_path = os.path.join('python_test/training', 'logs').replace('\\', '/')
PPO_path = os.path.join('python_test/training', 'python_test/models', 'PPO_CartPole-v1').replace('\\', '/')
os.makedirs(log_path, exist_ok=True)
os.makedirs(PPO_path, exist_ok=True)

# Create environment
env = DummyVecEnv([lambda: gym.make('CartPole-v1')])

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
# model.learn(total_timesteps=1000)

# model.save(PPO_path)
model = PPO.load(PPO_path + '.zip', env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")

for _ in range(5):
    state = State(env.reset())
    score = 0

    while not state.done:
        env.render()

        action, _ = model.predict(state.obs)
        print(f"Action: {action}")
        state = State(env.step(action))
        print(f'Info: {state.info}')
        score += state.reward

    print(f"Score: {score}")
