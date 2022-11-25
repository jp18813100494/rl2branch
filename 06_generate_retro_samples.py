# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Generates training samples for the imitation learning method of Gasse et al.  #                                                                     #
# Usage:                                                                        #
# python 03_generate_il_samples.py <type> -s <seed> -j <njobs>                  #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import glob
import gzip
import argparse
import pickle
import queue
import shutil
import threading
import numpy as np
import ecole
import utilities
import copy
from collections import namedtuple
from agent import TreeRecorder
from utilities import RetroBranching

class ExploreThenStrongBranch:
    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        probabilities = [1-self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model,done), True)
        else:
            return (self.pseudocosts_function.extract(model,done), False)

class PseudocostBranchingAgent:
    def __init__(self, name='pc'):
        self.name = name
        self.pc_branching_function = ecole.observation.Pseudocosts()

    def before_reset(self, model):
        self.pc_branching_function.before_reset(model)

    def extract(self, model, done):
        return self.pc_branching_function.extract(model, done)

    def action_select(self, action_set, model, done, **kwargs):
        scores = self.extract(model, done)
        action_idx = scores[action_set].argmax()
        return action_set[action_idx], action_idx




    
def send_orders(orders_queue, instances_path, seed, out_dir, stop_flag, mode='mdp',agent=None, sample_rate=1.0):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ---------- 
    orders_queue : queue.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    stop_flag: threading.Event
        A flag to tell the thread to stop.
    """
    # rng = np.random.RandomState(seed)
    instances = ecole.instance.FileGenerator(instances_path)
    episode = 0
    while not stop_flag.is_set():
        instance = next(instances)
        seed = rng.randint(2**32)
        orders_queue.put([episode, instance, seed, out_dir, agent,sample_rate, mode])
        episode += 1


def make_samples(in_queue, out_queue, stop_flag):
    """
    Worker loop: fetch an instance, run an episode and record samples.
    Parameters
    ----------
    in_queue : queue.Queue
        Input queue from which orders are received.
    out_queue : queue.Queue
        Output queue in which to send samples.
    stop_flag: threading.Event
        A flag to tell the thread to stop.
    """
    sample_counter = 0
    while not stop_flag.is_set():
        episode, instance, seed, out_dir, agent, sample_rate,mode= in_queue.get()
        observation_function = {"focus_node":ecole.observation.FocusNode(),
                                "node_observation": ecole.observation.NodeBipartite() }
        scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0,
                    'limits/time': time_limit, 'timing/clocktype': 2}
        agent = PseudocostBranchingAgent()
        env = ecole.environment.Branching(observation_function=observation_function,
                                          reward_function='default',
                                          information_function='default',
                                          scip_params=scip_parameters,
                                          )
        print(f"[w {threading.current_thread().name}] episode {episode}, seed {seed}, "f"processing instance '{instance.name}'...\n", end='')
        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance.name,
            'seed': seed,
        })
        # initialise custom reward object
        custom_reward = RetroBranching(debug_mode=False)
        # initialise an instance which is not pre-solved by SCIP

        env.seed(seed)
        custom_reward.before_reset(instance)
        agent.before_reset(instance)
        obs, action_set, reward, done, _ = env.reset(instance)
        _custom_reward = custom_reward.extract(env.model, done)

        prev_obs = obs
        if not done:
            focus_node_obs,node_observation = obs["focus_node"],obs["node_observation"]
            pre_state = utilities.extract_state(node_observation, action_set, focus_node_obs.number)
        original_transitions = []
        while not done:
            action, action_idx = agent.action_select(action_set, model=env.model, done=done)
            obs, action_set, reward, done, _ = env.step(action)
            _custom_reward = custom_reward.extract(env.model, done)

            if done:
                obs = prev_obs
                state = pre_state
            else:
                focus_node_obs,node_observation = obs["focus_node"],obs["node_observation"]
                state = utilities.extract_state(node_observation, action_set, focus_node_obs.number)
            if not stop_flag.is_set() and sample_rate > 0:
                original_transitions.append({'obs': prev_obs,
                                        'state': pre_state,
                                        'action': action,
                                        'action_idx': action_idx,
                                        'reward': None,
                                        'done': done,
                                        'next_obs': obs,
                                        'next_state': state}) 
                sample_counter += 1
            # update prev obs
            prev_obs = obs
            pre_state = state
        if _custom_reward:
            for retro_traj in _custom_reward:
                t = 0
                path_nodes = list(retro_traj.keys())
                for step_idx, reward in retro_traj.items():
                    transition = original_transitions[step_idx]
                    transition['reward'] = reward
                    if mode == 'retro':
                        if t < len(path_nodes)-1:
                            transition['next_state'] = original_transitions[path_nodes[t+1]]['state']
                        else:
                            transition['done'] = True
                    filename = f'{out_dir}/sample_{episode}_{step_idx}.pkl'
                    data = [transition['state'], transition['action'],transition['action_idx'],transition['reward'],transition['done'],transition['next_state']]
                    with gzip.open(filename, 'wb') as f:
                        pickle.dump({
                            'episode': episode,
                            'instance': instance.name,
                            'seed': seed,
                            'data': data,
                            }, f)
                    out_queue.put({
                        'type': 'sample',
                        'episode': episode,
                        'instance': instance.name,
                        'seed': seed,
                        'filename': filename,
                    })
                    t += 1
                    print(f"[w {threading.current_thread().name}] episode {episode} done, {step_idx} samples\n", end='')
        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance.name,
            'seed': seed,
        })


def collect_samples(instances_path, out_dir, rng, n_samples, n_jobs, mode):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-full strong' expert
    brancher.
    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """
    os.makedirs(out_dir, exist_ok=True)

    # start workers
    orders_queue = queue.Queue(maxsize=2*n_jobs)
    answers_queue = queue.SimpleQueue()

    tmp_samples_dir = f'{out_dir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # start dispatcher
    dispatcher_stop_flag = threading.Event()
    dispatcher = threading.Thread(
            target=send_orders,
            args=(orders_queue, instances_path, rng.randint(2**32), tmp_samples_dir, dispatcher_stop_flag, mode),
            daemon=True)
    dispatcher.start()

    workers = []
    workers_stop_flag = threading.Event()
    for i in range(n_jobs):
        p = threading.Thread(
                target=make_samples,
                args=(orders_queue, answers_queue, workers_stop_flag),
                daemon=True)
        workers.append(p)
        p.start()

    # record answers and write samples
    buffer = {}
    current_episode = 0
    i = 0
    in_buffer = 0
    while i < n_samples:
        sample = answers_queue.get()

        # add received sample to buffer
        if sample['type'] == 'start':
            buffer[sample['episode']] = []
        else:
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        # if any, write samples from current episode
        while current_episode in buffer and buffer[current_episode]:
            samples_to_write = buffer[current_episode]
            buffer[current_episode] = []

            for sample in samples_to_write:

                # if no more samples here, move to next episode
                if sample['type'] == 'done':
                    del buffer[current_episode]
                    current_episode += 1

                # else write sample
                else:
                    os.rename(sample['filename'], f'{out_dir}/sample_{i+1}.pkl')
                    in_buffer -= 1
                    i += 1
                    print(f"[m {threading.current_thread().name}] {i} / {n_samples} samples written, "
                          f"ep {sample['episode']} ({in_buffer} in buffer).\n", end='')

                    # early stop dispatcher
                    if in_buffer + i >= n_samples and dispatcher.is_alive():
                        dispatcher_stop_flag.set()
                        print(f"[m {threading.current_thread().name}] dispatcher stopped...\n", end='')

                    # as soon as enough samples are collected, stop
                    if i == n_samples:
                        buffer = {}
                        break

    # # stop all workers
    workers_stop_flag.set()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'ufacilities', 'indset', 'mknapsack'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument("--mode", type=str, default='mdp', choices=['mdp', 'tmdp+ObjLim', 'tmdp+DFS', 'retro', 'retro2'], help="Mode for branch env")
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    args = parser.parse_args()

    print(f"seed {args.seed}")

    train_size = 100000
    valid_size = 20000
    node_record_prob = 1.0
    time_limit = 3600

    if args.problem == 'setcover':
        instances_train = glob.glob('data/instances/setcover/train_400r_750c_0.05d/*.lp')
        instances_valid = glob.glob('data/instances/setcover/valid_400r_750c_0.05d/*.lp')
        instances_train_path = 'data/instances/setcover/train_400r_750c_0.05d'
        instances_valid_path = 'data/instances/setcover/valid_400r_750c_0.05d'
        out_dir = 'data/samples_{}/setcover/400r_750c_0.05d'.format(args.mode)
    elif args.problem == 'cauctions':
        instances_train = glob.glob('data/instances/cauctions/train_100_500/*.lp')
        instances_valid = glob.glob('data/instances/cauctions/valid_100_500/*.lp')
        instances_train_path = 'data/instances/cauctions/train_100_500'
        instances_valid_path = 'data/instances/cauctions/valid_100_500'
        out_dir = 'data/samples_{}/cauctions/100_500'.format(args.mode)

    elif args.problem == 'indset':
        instances_train = glob.glob('data/instances/indset/train_500_4/*.lp')
        instances_valid = glob.glob('data/instances/indset/valid_500_4/*.lp')
        instances_train_path = 'data/instances/indset/train_500_4'
        instances_valid_path = 'data/instances/indset/valid_500_4'
        out_dir = 'data/samples_{}/indset/500_4'.format(args.mode)

    elif args.problem == 'ufacilities':
        instances_train = glob.glob('data/instances/ufacilities/train_35_35_5/*.lp')
        instances_valid = glob.glob('data/instances/ufacilities/valid_35_35_5/*.lp')
        instances_train_path = 'data/instances/ufacilities/train_35_35_5'
        instances_valid_path = 'data/instances/ufacilities/valid_35_35_5'
        out_dir = 'data/samples_{}/ufacilities/35_35_5'.format(args.mode)
        time_limit = 600

    elif args.problem == 'mknapsack':
        instances_train = glob.glob('data/instances/mknapsack/train_100_6/*.lp')
        instances_valid = glob.glob('data/instances/mknapsack/valid_100_6/*.lp')
        instances_train_path = 'data/instances/mknapsack/train_100_6'
        instances_valid_path = 'data/instances/mknapsack/valid_100_6'
        out_dir = 'data/samples_{}/mknapsack/100_6'.format(args.mode)
        time_limit = 60

    else:
        raise NotImplementedError

    print(f"{len(instances_train)} train instances for {train_size} samples")
    print(f"{len(instances_valid)} validation instances for {valid_size} samples")

    # create output directory, throws an error if it already exists
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    collect_samples(instances_train_path, out_dir + '/train', rng, train_size,
                    args.njobs, mode=args.mode)

    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_valid_path, out_dir + '/valid', rng, valid_size,
                    args.njobs, mode=args.mode)
