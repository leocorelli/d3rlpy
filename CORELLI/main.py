from d3rlpy.datasets import get_atari
from d3rlpy.algos import DiscreteCQL
from d3rlpy.ope import DiscreteFQE
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer
from d3rlpy.metrics.scorer import true_q_scorer
from sklearn.model_selection import train_test_split

dataset, env = get_atari()
cql = DiscreteCQL(use_gpu=True, n_frames = 4, scaler = 'pixel')    # instantiate discrete CQL algrithm

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

'''
fqe = DiscreteFQE(algo=cql, use_gpu=True)
fqe.fit(
    train_episodes,
    eval_episodes=test_episodes,
    n_epochs=10,
    scorers={
        'estimated_q_values': initial_state_value_estimation_scorer,
        'soft_opc': soft_opc_scorer(70),
        'true_q_values': true_q_scorer},
    with_timestamp=False,
    experiment_name='DiscreteFQE_v0')
'''
