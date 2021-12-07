<p align="center"><img align="center" width="300px" src="assets/logo.png"></p>

# AIPI 530 Final Project: Using d3rlpy to perform offline deep reinforcement learning by Leo Corelli

d3rlpy is an offline deep reinforcement learning library for practitioners and researchers. In this project, I added to d3rlpy and used it to successfully build an offline deep reinforcement learning pipeline. After forking d3rlpy I did two things: 1) updated scorers.py to add a true Q value scorer function and 2) wrote my own script main.py, in which I successfully implemented and trained an agent to beat the cartpole-v0 task. I then evaluated my results using the off policy evaluation method of fitted Q evaluation, and plotted my results along the way.

Below is an example of how to implement d3rlpy to build an offline deep reinforcement learning pipeline:

```py
import d3rlpy
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

## Installation
d3rlpy supports Linux, macOS and Windows.

1. Clone repository: ```$ git clone https://github.com/leocorelli/d3rlpy.git```
2. Install requirements: ```$ pip install Cython numpy``` and ```$ pip install -e```
3. Run main.py (in CORELLI folder)

## Results
<p align="center"><img align="center" width="700px" src="https://github.com/leocorelli/d3rlpy/blob/master/CORELLI/average_reward.png"></p>
<p align="center"><img align="center" width="700px" src="https://github.com/leocorelli/d3rlpy/blob/master/CORELLI/true_q.png"></p>
<p align="center"><img align="center" width="700px" src="https://github.com/leocorelli/d3rlpy/blob/master/CORELLI/estimated_q.png"></p>
<p align="center"><img align="center" width="700px" src="https://github.com/leocorelli/d3rlpy/blob/master/CORELLI/fqe.png"></p>




## tutorials
Try a cartpole example on Google Colaboratory!

- offline RL tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takuseno/d3rlpy/blob/master/tutorials/cartpole.ipynb)
- online RL tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takuseno/d3rlpy/blob/master/tutorials/online.ipynb)

## contributions
Any kind of contribution to d3rlpy would be highly appreciated!
Please check the [contribution guide](CONTRIBUTING.md).

The release planning can be checked at [milestones](https://github.com/takuseno/d3rlpy/milestones).

## community
| Channel | Link |
|:-|:-|
| Chat | [Gitter](https://gitter.im/d3rlpy/d3rlpy) |
| Issues | [GitHub Issues](https://github.com/takuseno/d3rlpy/issues) |

## family projects
| Project | Description |
|:-:|:-|
| [d4rl-pybullet](https://github.com/takuseno/d4rl-pybullet) | An offline RL datasets of PyBullet tasks |
| [d4rl-atari](https://github.com/takuseno/d4rl-atari) | A d4rl-style library of Google's Atari 2600 datasets |
| [MINERVA](https://github.com/takuseno/minerva) | An out-of-the-box GUI tool for offline RL |

## roadmap
The roadmap to the future release is available in [ROADMAP.md](ROADMAP.md).

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

## acknowledgement
This work is supported by Information-technology Promotion Agency, Japan
(IPA), Exploratory IT Human Resources Project (MITOU Program) in the fiscal
year 2020.
