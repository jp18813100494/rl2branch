import os
import json
import gzip
from tkinter import N
import ecole
import pickle
import logging
import threading
import glog
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from datetime import datetime
from pathlib import Path
from scipy.stats.mstats import gmean
from iql.iql_agent import save
import random



def log(str, logfile=None):
    str = f'[{datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

class State(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_index, edge_attr, variable_features,
                 action_set, action_set_size, node_id):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.variable_features = variable_features
        self.action_set = action_set
        self.action_set_size = action_set_size
        self.node_id = node_id

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'action_set':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)

    def to(self, device):
        """
        Current version is inplace, which is incoherent with how pytorch's to() function works.
        This overloads it.
        """
        cuda_values = {key: self[key].to(device) if isinstance(self[key], torch.Tensor) else self[key]
                        for key in self.keys}
        return State(**cuda_values)

class Transition(torch_geometric.data.Data):
    def __init__(self, state, action=None, cum_nnodes=None):
        super().__init__()
        self.constraint_features = state.constraint_features
        self.edge_index = state.edge_index
        self.edge_attr = state.edge_attr
        self.variable_features = state.variable_features
        self.action_set = state.action_set
        self.action_set_size = state.action_set_size
        self.node_id = state.node_id
        self.num_nodes = state.num_nodes
        assert self.edge_index.max()<self.variable_features.shape[0]
        self.action = torch.LongTensor(np.array([action],dtype=np.int32))
        self.cum_nnodes = cum_nnodes
        self.returns = None

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'action_set':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)

    def to(self, device):
        """
        Current version is inplace, which is incoherent with how pytorch's to() function works.
        This overloads it.
        """
        cuda_values = {key: self[key].to(device) if isinstance(self[key], torch.Tensor) else self[key]
                        for key in self.keys}
        return Transition(**cuda_values)

class FullTransition(Transition):
    def __init__(self, state, action,scores,reward,done,next_state,cum_nnodes=None):
        super().__init__(state,action,cum_nnodes)
        action_idx = scores[self.action_set].argmax()
        self.action_idx = torch.LongTensor(np.array([action_idx],dtype=np.int32))
        self.scores = torch.LongTensor(scores)
        self.reward = torch.FloatTensor(np.expand_dims(-reward, axis=-1))
        self.done = torch.FloatTensor(np.expand_dims(int(done), axis=-1))

        self.constraint_features_n = next_state.constraint_features
        self.edge_index_n = next_state.edge_index
        self.edge_attr_n = next_state.edge_attr
        self.variable_features_n = next_state.variable_features
        self.action_set_n = next_state.action_set
        self.action_set_n_size = next_state.action_set_size
        self.node_id_n = next_state.node_id
        self.num_nodes_n = next_state.num_nodes
        assert self.edge_index_n.max()<self.variable_features_n.shape[0]

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        if key == 'edge_index_n':
            return torch.tensor([[self.constraint_features_n.size(0)], [self.variable_features_n.size(0)]])
        elif key == 'action_set':
            return self.variable_features.size(0)
        elif key == 'action_set_n':
            return self.variable_features_n.size(0)
        else:
            return super().__inc__(key, value)

    def to(self, device):
        """
        Current version is inplace, which is incoherent with how pytorch's to() function works.
        This overloads it.
        """
        cuda_values = {key: self[key].to(device) if isinstance(self[key], torch.Tensor) else self[key]
                        for key in self.keys}
        return FullTransition(**cuda_values)

def BuildFullTransition(data_files):
    transitions = []
    for sample_file in data_files:
        with gzip.open(sample_file, 'rb') as f:
            sample = pickle.load(f)
        state, action, scores, reward, done, next_state = sample['data']
        fulltransition = FullTransition(state,action,scores,reward,done,next_state)
        transitions.append(fulltransition)
    random.shuffle(transitions)
    return transitions

def extract_state(observation, action_set, node_id):
    constraint_features = torch.FloatTensor(observation.row_features)
    edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int64))
    edge_attr = torch.FloatTensor(np.expand_dims(observation.edge_features.values, axis=-1))
    variable_features = torch.FloatTensor(observation.column_features)
    action_set = torch.LongTensor(np.array(action_set, dtype=np.int64))
    action_set_size = action_set.shape[0]
    node_id = node_id

    state = State(constraint_features, edge_index, edge_attr, variable_features, action_set, action_set_size, node_id)
    state.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
    return state


def pad_tensor(input_, pad_sizes, value=False, pad_value=-1e8):
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    if value:
        output = torch.stack([torch.mean(slice_) for slice_ in output], dim=0).unsqueeze(-1)
    else:
        output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output


class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, candidate_choice, candidate_scores):
        super(BipartiteNodeData,self).__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = len(candidates)
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, sample_files):
        super(GraphDataset,self).__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample['data']

        constraint_features, (edge_indices, edge_features), variable_features = sample_observation
        constraint_features = torch.FloatTensor(constraint_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
        variable_features = torch.FloatTensor(variable_features)

        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_choice = torch.where(candidates == sample_action)[0][0]  # action index relative to candidates
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, candidate_choice, candidate_scores)
        # print(graph)
        graph.num_nodes = constraint_features.shape[0]+variable_features.shape[0]
        return graph

# Not used in current version
class ExtendGraphDataset(torch_geometric.data.Dataset):
    def __init__(self, sample_files):
        super(ExtendGraphDataset,self).__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        state, action, scores, reward, done, next_state = sample['data']
        graph = extract_graph(state,action,scores)
        #define action,reward,terminals
        action = torch.FloatTensor(np.expand_dims(action, axis=-1))
        reward = torch.FloatTensor(np.expand_dims(reward, axis=-1))
        terminal = torch.FloatTensor(np.expand_dims(done.int(), axis=-1))
        graph_next = extract_graph(next_state)
        return (graph,action,reward,graph_next,terminal)

def extract_graph(state,action=None,scores=None,):
    #define state-graph
    constraint_features = torch.FloatTensor(state['constraint_features'])
    edge_indices = torch.LongTensor(state['edge_indices'].astype(np.int32))
    edge_features = torch.FloatTensor(np.expand_dims(state['edge_features'], axis=-1))
    variable_features = torch.FloatTensor(state['variable_features'])

    candidates = torch.LongTensor(np.array(state['action_set'], dtype=np.int32))
    candidate_choice = torch.where(candidates == action)[0][0]  # action index relative to candidates
    candidate_scores = torch.FloatTensor([scores[j] for j in candidates])

    graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                candidates, candidate_choice, candidate_scores)
    graph.num_nodes = constraint_features.shape[0]+variable_features.shape[0]
    return graph


class Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch =+1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs == self.patience:
            self._reduce_lr(self.last_epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class FormatterWithHeader(logging.Formatter):
    """
    From
    https://stackoverflow.com/questions/33468174/write-header-to-a-python-log-file-but-only-if-a-record-gets-written
    """
    def __init__(self, header, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.header = header
        self.format = self.first_line_format

    def first_line_format(self, record):
        self.format = super().format
        return self.header + "\n" + self.format(record)


def configure_logging(header=""):
    os.makedirs("logs/", exist_ok=True)
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    output_file = f"logs/{current_date}--{current_time.replace(':','.')}.log"
    logging_header = (
    f"rl2branch log\n"
    f"-------------\n"
    f"Training started on {current_date} at {current_time}\n"
    )

    logger = logging.getLogger("rl2branch")
    logger.setLevel(logging.DEBUG)

    formatter = FormatterWithHeader(header=header,
                                    fmt='[%(asctime)s %(levelname)-8s]  %(threadName)-12s  %(message)s',
                                    datefmt='%H:%M:%S')

    handler_file = logging.FileHandler(output_file, 'w', 'utf-8')
    handler_file.setLevel(logging.DEBUG)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)

    handler_console = logging.StreamHandler()
    handler_console.setLevel(logging.DEBUG)
    handler_console.setFormatter(formatter)
    logger.addHandler(handler_console)
    return logger

def evaluate(env, policy, eval_runs,logger): 
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    stats = []
    for i in range(eval_runs):
        state = env.reset()
        iter_count = 0
        rewards = 0
        while True:
            action = policy.get_action(state, eval=True)
            state, reward, done, info = env.step(action)
            rewards += reward
            iter_count += 1
            if done:
                break
        logger.info(f'Eval Task:{i}, {env.instance},Total Reward:{rewards}, NNodes:{info["nnodes"]},Lpiters:{info["lpiters"]},Time:{info["time"]}')
        stats.append({'task':env.instance,'info':info})
        reward_batch.append(rewards)
    return reward_batch, stats


def evaluate_single(i, queue, env, policy,logger): 
    """
    Makes an evaluation run with the current policy
    """

    state = env.reset()
    count = 0
    rewards = 0
    while True:
        action = policy.get_action(state, eval=True)
        state, reward, done, info = env.step(action)
        rewards += reward
        count += 1
        if done:
            break
    logger.info(f'Eval Task:{i}, {env.instance},Total Reward:{rewards}, Episode:{count} NNodes:{info["nnodes"]},Lpiters:{info["lpiters"]},Time:{info["time"]}')
    stat = {'task':env.instance,'info':info}
    queue.put((rewards,stat,count))

def evaluate_parellel(env, policy, config, logger):
    """
    Make an evaluation run with parellerl agents
    """
    #TODO:
    eval_run, num_workers = config['eval_run'], config['num_workers']
    rewards, stats, ep_lens =[], [],[]
    assert eval_run>=num_workers
    cnt = eval_run//num_workers
    for i in range(cnt):
        info_queue = mp.Queue()
        processes = []
        for j in range(num_workers):
            process = Process(target=evaluate_single, args=(i*num_workers+j, info_queue, env, policy, logger))
            process.daemon = True
            processes.append(process)
        [p.start() for p in processes]
        while True:
            total_reward, stat,ep_len = info_queue.get()
            if total_reward is not None:
                rewards.append(total_reward)
                stats.append(stat)
                ep_lens.append(ep_len)
            else:
                break
        [p.join() for p in processes]
    return rewards, stats, ep_lens

def wandb_eval_log(epoch, agent, wandb, wandb_data, v_stats, v_reward, config, logger,stat='offline'):
    v_nnodess = [s['info']['nnodes'] for s in v_stats]
    v_lpiterss = [s['info']['lpiters'] for s in v_stats]
    v_times = [s['info']['time'] for s in v_stats]

    wandb_data.update({
        'valid_reward':np.mean(v_reward),
        'valid_nnodes_g': gmean(np.asarray(v_nnodess) + 1) - 1,
        'valid_nnodes': np.mean(v_nnodess),
        'valid_nnodes_std': np.std(v_nnodess),
        'valid_nnodes_max': np.amax(v_nnodess),
        'valid_nnodes_min': np.amin(v_nnodess),
        'valid_time': np.mean(v_times),
        'valid_lpiters': np.mean(v_lpiterss),
    })
    if epoch == 0:
        v_nnodes_0 = wandb_data['valid_nnodes'] if wandb_data['valid_nnodes'] != 0 else 1
        v_nnodes_g_0 = wandb_data['valid_nnodes_g'] if wandb_data['valid_nnodes_g']!= 0 else 1
        config["v_nnodes_0"] = v_nnodes_0
        config["v_nnodes_g_0"] = v_nnodes_g_0
    wandb_data.update({
        'valid_nnodes_norm': wandb_data['valid_nnodes'] / config["v_nnodes_0"],
        'valid_nnodes_g_norm': wandb_data['valid_nnodes_g'] / config["v_nnodes_g_0"],
    })

    if wandb_data['valid_nnodes_g'] < config['best_tree_size']:
        config['best_tree_size'] = wandb_data['valid_nnodes_g']
        logger.info('Best parameters so far (1-shifted geometric mean), saving model.')
        if config['wandb']:
            save(config, model=agent, wandb=wandb ,stat=stat)
    return wandb_data
