import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
import tensorflow as tf

from load_policy import load_policy
from env.gym_env import GymEnv
from utility.utils import check_make_dir
from utility.debug_tools import pwc


def parse_cmd_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--video_path', '-v', action='store_true')
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=1000)
    parser.add_argument('--num_rollouts', type=int, default=1000,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    return args

def main():
    args = parse_cmd_args()

    print('loading and building expert policy')
    policy_fn = load_policy(args.expert_policy_file)
    print('loaded and built')

    env_args = dict(name=args.envname)

    if args.video_path:
        env_args['video_path'] = f'experts/data/{env_args['name']}'
    env_args['max_episode_steps'] = args.max_timesteps

    env = GymEnv(env_args)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        max_steps = args.max_timesteps or env.max_episode_steps

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: 
                    print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        path = os.path.join('experts', 'data', args.envname + '.pkl')
        check_make_dir(path)
        with open(path, 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
