import d3rlpy
from d3rlpy.datasets import get_cartpole

dataset, env = get_cartpole()

cql = d3rlpy.algos.DiscreteCQL(use_gpu = True)    # instantiate discrete CQL algrithm 
cql.fit(dataset, n_steps = 10000)   # train (offline)
# actions = cql.predict(x)    # generate actions

