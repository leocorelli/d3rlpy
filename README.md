<p align="center"><img align="center" width="300px" src="assets/logo.png"></p>

# AIPI 530 Final Project: Using d3rlpy to perform offline deep reinforcement learning by Leo Corelli

d3rlpy is an offline deep reinforcement learning library for practitioners and researchers. In this project, I added to d3rlpy and used it to successfully build an offline deep reinforcement learning pipeline. After forking d3rlpy I did two things: 1) updated scorers.py to add a true Q value scorer function and 2) wrote my own script main.py, in which I successfully implemented and trained an agent to beat the cartpole-v0 task. I then evaluated my results using the off policy evaluation method of fitted Q evaluation, and plotted my results along the way.

Below is an example of how to implement d3rlpy to build an offline deep reinforcement learning pipeline:

```py
from d3rlpy.datasets import get_cartpole
from d3rlpy.algos import DiscreteCQL
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import true_q_scorer
from sklearn.model_selection import train_test_split


dataset, env = get_cartpole() 
cql = DiscreteCQL(use_gpu=True)   

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)
cql.fit(
    train_episodes,
    eval_episodes=test_episodes,
    n_epochs=10,
    scorers={
        'average_reward': evaluate_on_environment(env),
        'estimated_q_values': initial_state_value_estimation_scorer,
        'true_q_values': true_q_scorer},
    with_timestamp=False,
    experiment_name='DiscreteCQL_v0')
```

- Original documentation: https://d3rlpy.readthedocs.io
- Original paper: https://arxiv.org/abs/2111.03788
- Original repository: https://github.com/takuseno/d3rlpy

## installation
d3rlpy supports Linux, macOS and Windows.

1. Clone repository: ```$ git clone https://github.com/leocorelli/d3rlpy.git```
2. Install requirements: ```$ pip install Cython numpy``` and ```$ pip install -e .```

## get started
1. Run ```main.py``` in ```/d3rlpy/CORELLI```
2. Logs:
    - Average reward: ```/d3rlpy_logs/DiscreteCQL_v0/average_reward.csv```
    - True Q value: ```/d3rlpy_logs/DiscreteCQL_v0/true_q_values.csv```
    - Estimated Q value: ```/d3rlpy_logs/DiscreteCQL_v0/estimated_q_values.csv```
    - FQE True Q value: ```/d3rlpy_logs/DiscreteFQE_v0/true_q_values.csv```
    - FQE Estimated Q value: ```/d3rlpy_logs/DiscreteFQE_v0/estimated_q_values.csv```

The general format of the offline reinforcement learning pipeline implemented in main.py can be applied to more than just the cartpole-v0 environment. For a full list of available datasets, please refer to /d3rlpy/datasets.

## results
In this project I used a discrete CQL algorithm to train an agent to play the cartpole game. I then evaluated my policy using off policy evaluation (OPE) by performing fitted q evaluation (FQE). The results of my experiments can be found below. *Note: the maximum reward in cartpole-v0 is 200.0*

<p align="center"><img align="center" width="700px" src="https://github.com/leocorelli/d3rlpy/blob/master/CORELLI/average_reward.png"></p>
<p align="center"><img align="center" width="700px" src="https://github.com/leocorelli/d3rlpy/blob/master/CORELLI/true_q.png"></p>
<p align="center"><img align="center" width="700px" src="https://github.com/leocorelli/d3rlpy/blob/master/CORELLI/estimated_q.png"></p>
<p align="center"><img align="center" width="700px" src="https://github.com/leocorelli/d3rlpy/blob/master/CORELLI/fqe.png"></p>

## supported algorithms
| algorithm | discrete control | continuous control | offline RL? |
|:-|:-:|:-:|:-:|
| Behavior Cloning (supervised learning) | :white_check_mark: | :white_check_mark: | |
| [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236) | :white_check_mark: | :no_entry: | |
| [Double DQN](https://arxiv.org/abs/1509.06461) | :white_check_mark: | :no_entry: | |
| [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) | :no_entry: | :white_check_mark: | |
| [Twin Delayed Deep Deterministic Policy Gradients (TD3)](https://arxiv.org/abs/1802.09477) | :no_entry: | :white_check_mark: | |
| [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1812.05905) | :white_check_mark: | :white_check_mark: | |
| [Batch Constrained Q-learning (BCQ)](https://arxiv.org/abs/1812.02900) | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [Bootstrapping Error Accumulation Reduction (BEAR)](https://arxiv.org/abs/1906.00949) | :no_entry: | :white_check_mark: | :white_check_mark: |
| [Advantage-Weighted Regression (AWR)](https://arxiv.org/abs/1910.00177) | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [Advantage Weighted Actor-Critic (AWAC)](https://arxiv.org/abs/2006.09359) | :no_entry: | :white_check_mark: | :white_check_mark: |
| [Critic Reguralized Regression (CRR)](https://arxiv.org/abs/2006.15134) | :no_entry: | :white_check_mark: | :white_check_mark: |
| [Policy in Latent Action Space (PLAS)](https://arxiv.org/abs/2011.07213) | :no_entry: | :white_check_mark: | :white_check_mark: |
| [TD3+BC](https://arxiv.org/abs/2106.06860) | :no_entry: | :white_check_mark: | :white_check_mark: |
| [Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169) | :no_entry: | :white_check_mark: | :white_check_mark: |

## supported Q functions
- [x] standard Q function
- [x] [Quantile Regression](https://arxiv.org/abs/1710.10044)
- [x] [Implicit Quantile Network](https://arxiv.org/abs/1806.06923)

## experimental features
- Model-based Algorithms
  - [Model-based Offline Policy Optimization (MOPO)](https://arxiv.org/abs/2005.13239)
  - [Conservative Offline Model-Based Policy Optimization (COMBO)](https://arxiv.org/abs/2102.08363)
- Q-functions
  - [Fully parametrized Quantile Function](https://arxiv.org/abs/1911.02140) (experimental)

## tutorials
Try a cartpole example on Google Colaboratory!

- offline RL tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takuseno/d3rlpy/blob/master/tutorials/cartpole.ipynb)
- online RL tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takuseno/d3rlpy/blob/master/tutorials/online.ipynb)


## citation
The paper is available [here](https://arxiv.org/abs/2111.03788).
```
@InProceedings{seno2021d3rlpy,
  author = {Takuma Seno, Michita Imai},
  title = {d3rlpy: An Offline Deep Reinforcement Library},
  booktitle = {NeurIPS 2021 Offline Reinforcement Learning Workshop},
  month = {December},
  year = {2021}
}
```

This repo was originally forked from user takuseno on GitHub. Tremendous thank you to them for all of their excellent work on this repository. All of my original files can be found in the CORELLI folder. In addition to files found here, I added a single function in scorer.py in d3rlpy/metrics in order to calculate the true q value at the end of each training epoch. The code for my true_q_scorer function was sourced from td_error_scorer also found in the scorer.py file, which was written by the original creator of this forked repository.

## acknowledgement
This work is supported by Information-technology Promotion Agency, Japan
(IPA), Exploratory IT Human Resources Project (MITOU Program) in the fiscal
year 2020.
