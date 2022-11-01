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
from collections import namedtuple
from agent import TreeRecorder


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


def send_orders(orders_queue, instances, seed, query_expert_prob, time_limit, out_dir, stop_flag,mode='mdp',agent=None, sample_rate=1.0):
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
    rng = np.random.RandomState(seed)

    episode = 0
    while not stop_flag.is_set():
        instance = rng.choice(instances)
        seed = rng.randint(2**32)
        orders_queue.put([episode, instance, seed, query_expert_prob, time_limit, out_dir, agent,sample_rate, mode])
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
        episode, instance, seed, query_expert_prob, time_limit, out_dir, agent, sample_rate, mode= in_queue.get()
        scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0,
                           'limits/time': time_limit, 'timing/clocktype': 2}
        observation_function = { "scores": ExploreThenStrongBranch(expert_probability=query_expert_prob),
                                "focus_node":ecole.observation.FocusNode(),
                                "node_observation": ecole.observation.NodeBipartite() }
        # reward_function= ecole.reward.NNodes()
        reward_function= {
            "cum_nnodes": ecole.reward.NNodes().cumsum(),
            "cur_nnodes": ecole.reward.NNodes()
        }
        env = ecole.environment.Branching(observation_function=observation_function,reward_function=reward_function,
                                          scip_params=scip_parameters, pseudo_candidates=True)
        print(f"[w {threading.current_thread().name}] episode {episode}, seed {seed}, "
              f"processing instance '{instance}'...\n", end='')
        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })
        if sample_rate > 0:
            tree_recorder = TreeRecorder()
        rng = np.random.RandomState(seed)
        datas = dict()
        env.seed(seed)
        observation, action_set, reward, done, _ = env.reset(instance)
        while not done:
            scores, scores_are_expert = observation["scores"]
            focus_node_obs = observation["focus_node"]
            node_observation = observation["node_observation"]
            cum_nnodes, cur_nnodes = reward['cum_nnodes'],reward['cur_nnodes']
            state = utilities.extract_state(node_observation, action_set, focus_node_obs.number)
            if agent is None:
                action_idx = scores[action_set].argmax()
                action = action_set[action_idx]
            else:
                action_idx = agent.get_action(state,eval=True)
                action = action_set[action_idx]
            if scores_are_expert and not stop_flag.is_set():
                if sample_rate > 0:
                    tree_recorder.record_branching_decision(focus_node_obs)
                    keep_sample = rng.rand() < sample_rate
                    if keep_sample:
                        datas[focus_node_obs.number] ={
                                    'episode':episode,
                                    'sample_counter':sample_counter,
                                    "state":state,
                                    'action':action,
                                    'action_idx':action_idx,
                                    'scores':scores,
                                    'cum_nnodes':cum_nnodes,
                                    'returns':None,
                                    'reward':None,
                                    'next_state':[None,None],
                                    'done':done
                                }  
                sample_counter += 1
            try:
                observation, action_set, reward, done, _ = env.step(action)
            except Exception as e:
                done = True
                with open("error_log.txt","a") as f:
                    f.write(f"Error occurred solving {instance} with seed {seed}\n")
                    f.write(f"{e}\n")
        if datas:
            # post-process the collected samples (credit assignment) from strong branch policy
            if sample_rate > 0:
                datas = tree_recorder.calculate_subtree_sizes_next_state(datas,mode)
            for key,value in datas.items():
                episode = value['episode']
                sample_counter = value['sample_counter']
                state = value['state']
                action = value['action']
                action_idx = value['action_idx']
                scores = value['scores']
                cum_nnodes = value['cum_nnodes']
                reward = value['reward']
                next_state = value['next_state']
                done = value['done']
                filename = f'{out_dir}/sample_{episode}_{sample_counter}.pkl'
                data = [state, action, action_idx, scores, reward,  done, next_state]
                with gzip.open(filename, 'wb') as f:
                    pickle.dump({
                        'episode': episode,
                        'instance': instance,
                        'seed': seed,
                        'data': data,
                        }, f)
                out_queue.put({
                    'type': 'sample',
                    'episode': episode,
                    'instance': instance,
                    'seed': seed,
                    'filename': filename,
                })
                print(f"[w {threading.current_thread().name}] episode {episode} done, {sample_counter} samples\n", end='')
        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })


def collect_samples(instances, out_dir, rng, n_samples, n_jobs,
                    query_expert_prob, time_limit, mode):
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
            args=(orders_queue, instances, rng.randint(2**32), query_expert_prob,
                  time_limit, tmp_samples_dir, dispatcher_stop_flag,mode),
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
    parser.add_argument("--mode", type=str, default='mdp', choices=['mdp', 'tmdp+ObjLim', 'tmdp+DFS'], help="Mode for branch env")
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
        out_dir = 'data/samples_{}/setcover/400r_750c_0.05d'.format(args.mode)

    elif args.problem == 'cauctions':
        instances_train = glob.glob('data/instances/cauctions/train_100_500/*.lp')
        instances_valid = glob.glob('data/instances/cauctions/valid_100_500/*.lp')
        out_dir = 'data/samples_{}/cauctions/100_500'.format(args.mode)

    elif args.problem == 'indset':
        instances_train = glob.glob('data/instances/indset/train_500_4/*.lp')
        instances_valid = glob.glob('data/instances/indset/valid_500_4/*.lp')
        out_dir = 'data/samples_{}/indset/500_4'.format(args.mode)

    elif args.problem == 'ufacilities':
        instances_train = glob.glob('data/instances/ufacilities/train_35_35_5/*.lp')
        instances_valid = glob.glob('data/instances/ufacilities/valid_35_35_5/*.lp')
        out_dir = 'data/samples_{}/ufacilities/35_35_5'.format(args.mode)
        time_limit = 600

    elif args.problem == 'mknapsack':
        instances_train = glob.glob('data/instances/mknapsack/train_100_6/*.lp')
        instances_valid = glob.glob('data/instances/mknapsack/valid_100_6/*.lp')
        out_dir = 'data/samples_{}/mknapsack/100_6'.format(args.mode)
        time_limit = 60

    else:
        raise NotImplementedError

    print(f"{len(instances_train)} train instances for {train_size} samples")
    print(f"{len(instances_valid)} validation instances for {valid_size} samples")

    # create output directory, throws an error if it already exists
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    collect_samples(instances_train, out_dir + '/train', rng, train_size,
                    args.njobs, query_expert_prob=node_record_prob,
                    time_limit=time_limit, mode=args.mode)

    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_valid, out_dir + '/valid', rng, valid_size,
                    args.njobs, query_expert_prob=node_record_prob,
                    time_limit=time_limit, mode=args.mode)
