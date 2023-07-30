# inhibit-selfish-rl

Studying and comparing approaches for inhibiting selfish behaviour of RL-based agents, in environments which encourage it.

This codebase was used for my final year dissertation, as part of my BSc CompSci degree in 2022/23.

## Description

Four multi-agent environments are evaluated, where two agents with not-relating or opposing goals exist.

The optimal policy for agent A is not aligned with agent B's goals.

This project compares three approaches for inhibiting selfish behaviour:

- Add B's reward function to A's reward function, with some multiplier
- Devise custom, per-environment, heuristics for reducing selfish behaviour
- Use Inverse Reinforcement Learning to approximate B's goals, then add them to A's reward function

## More info

If you want to learn more, you can check out this [poster](https://github.com/cemiu/inhibit-selfish-rl/blob/main/results/poster.pdf), or [presentation](https://github.com/cemiu/inhibit-selfish-rl/blob/main/results/presentation.key) (Apple Keynotes), [presentation](https://github.com/cemiu/inhibit-selfish-rl/blob/main/results/presentation_no_videos.pdf) (PDF, but without videos), or just read the [unnecessarily long dissertation](https://github.com/cemiu/inhibit-selfish-rl/blob/main/results/dissertation.pdf) I wrote about it.


## Setup

At the time of creation, these commands worked for setting up all dependencies. With SB3 moving to gymnasium & imitation planning to later do the same, this is unlikely to work anymore.

pip install opencv-python matplotlib pandas seaborn tensorboard numpy imitation jupyter
pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests

imitation=1.1.1
