import gymnasium as gym
from util.state_util import State

field = ['F' * 4] * 4
# noinspection SpellCheckingInspection
field[0] = 'SFFF'
# noinspection SpellCheckingInspection
field[3] = 'FFFG'

env = gym.make('FrozenLake-v1', render_mode='rgb_array', desc=None)

env.reset()

observation_stack = [0]

limit = 1000000

for i in range(limit):
    env.render()
    action = env.action_space.sample()
    # observation, reward, done, trunc, info = env.step(action)
    obs = State(env.step(action))
    # print(obs.observation)
    observation_stack.append(obs.obs)

    # print(obs)

    if obs.reward > 0:
        # print('Reward:', obs.reward)
        print('Obs stack:', observation_stack)

    if obs.done or len(observation_stack) > 100:
        env.reset()
        observation_stack = [0]

env.close()
