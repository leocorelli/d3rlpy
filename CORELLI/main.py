import d3rlpy

dataset, env = d3rlpy.datasets.get_cartpole()

cql = d3rlpy.algos.DiscreteCQL(use_gpu = True)    # instantiate 
cql.fit(dataset, n_steps = 100)   # train (offline)

#actions = cql.predict(x)    # generate actions

