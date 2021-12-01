import d3rlpy
from d3rlpy.datasets import get_cartpole
from d3rlpy.algos import DiscreteCQL
from sklearn.model_selection import train_test_split

dataset, env = get_cartpole() 
cql = DiscreteCQL(use_gpu = True)    # instantiate discrete CQL algrithm 

train_episodes, test_episodes = train_test_split(dataset, test_size = 0.2)

#cql.fit(dataset, n_steps = 10000)   # train (offline)
cql.fit(train_episodes, eval_episodes=test_episodes, n_epochs=1, n_steps = 10000)
# actions = cql.predict(x)    # generate actions

