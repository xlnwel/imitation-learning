import os
import csv
from copy import deepcopy
import pickle
import numpy as np

from utility.utils import set_global_seed, check_make_dir
from utility.debug_tools import pwc
from algo.dagger.agent import Agent
from experts.policy import ExpertPolicy


def run_trajectory(agent, env, fn):
    """ run a trajectory, fn is a function executed after each environment step """
    state = env.reset()
    for i in range(env.max_episode_steps):
        action = agent.act(state)
        next_state, _, done, _ = env.step(action)
        fn(state, action)
        state = next_state
        if done:
            break

    return i, env.get_score()

def run_expert(expert_policy, env, env_name, buffer, num_steps):
    def fn(state, action):
        buffer.add(state, action)

    i = 0
    scores = []
    while i <= num_steps:
        k, score = run_trajectory(expert_policy, env, fn)
        print(f'\rSteps {i}: score={score}\t avg_score={np.mean(scores)}', end='')
        i += k + 1
        scores.append(score)
    
    print()
    mean = np.mean(scores)
    std = np.std(scores)
    pwc(f'Average Score: {mean}')
    pwc(f'Average Mean: {std}')
    w = csv.writer(open(buffer.filename[:-3] + 'csv', 'w'))
    w.writerow(['score_mean', mean])
    w.writerow(['score_std', std])
    with open(buffer.filename, 'wb') as f:
        pickle.dump(buffer.memory, f, pickle.HIGHEST_PROTOCOL)

    pwc('Expert trajectories are all collected')

def train(agent, algo_name, env_name):
    expert_file = f'experts/{env_name}.pkl'
    expert_policy = ExpertPolicy(expert_file)
    def fn(state, action):
        if algo_name == 'dagger':
            action = expert_policy.act(state[None, :])
            agent.buffer.add(state, action)

    if not agent.buffer.full:
        run_expert(expert_policy, agent.env, env_name, agent.buffer, agent.buffer.capacity - agent.buffer.idx)

    for i in range(agent.args['num_epochs']):
        agent.learn(i)
        if i % 100 == 0:
            scores = []
            for _ in range(10):
                _, score = run_trajectory(agent, agent.env, fn)
                scores.append(score)
            
            if hasattr(agent, 'stats'):
                agent.record_stats(score_mean=np.mean(scores), score_std=np.std(scores))
            log_info = {
                'ModelName': f'{algo_name}-{env_name}',
                'ScoreMean': np.mean(scores),
                'ScoreStd': np.std(scores)
            }
            [agent.log_tabular(k, v) for k, v in log_info.items()]
            agent.dump_tabular(print_terminal_info=True)

def main(env_args, agent_args, buffer_args, render=False):
    set_global_seed()

    algorithm = agent_args['algorithm']
    buffer_args['filename'] = f'experts/data/{env_args["name"]}.pkl'
    agent = Agent(agent_args['algorithm'], agent_args, env_args, buffer_args, log_tensorboard=True, log_stats=True, save=True)

    train(agent, algorithm, env_args['name'])
