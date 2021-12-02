from d3rlpy.datasets import get_cartpole
from d3rlpy.algos import DiscreteCQL
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from sklearn.model_selection import train_test_split


dataset, env = get_cartpole() 
cql = DiscreteCQL(use_gpu = True)    # instantiate discrete CQL algrithm 

train_episodes, test_episodes = train_test_split(dataset, test_size = 0.2)

#cql.fit(dataset, n_steps = 10000)   # train (offline)
cql.fit(train_episodes, eval_episodes=test_episodes, n_epochs = 25, scorers = {'average_reward': evaluate_on_environment(env),'estimated_q_values':initial_state_value_estimation_scorer}, with_timestamp=False, experiment_name = 'DiscreteCQL_v0')
# actions = cql.predict(x)    # generate actions