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

This repo was originally forked from user takuseno on GitHub. Tremendous thank you to them for all of their excellent work on this repository. All of my original files can be found in the CORELLI folder. In addition to files found here, I added a single function in scorer.py in d3rlpy/metrics in order to calculate the true q value at the end of each training epoch.

## acknowledgement
This work is supported by Information-technology Promotion Agency, Japan
(IPA), Exploratory IT Human Resources Project (MITOU Program) in the fiscal
year 2020.
