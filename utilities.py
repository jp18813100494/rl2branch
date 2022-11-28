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
import random
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from retro_branching.utils import SearchTree, seed_stochastic_modules_globally


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
        # assert self.edge_index.max()<self.variable_features.shape[0]
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
    def __init__(self, state, action, action_idx, reward, done, next_state, cum_nnodes=None):
        super().__init__(state,action,cum_nnodes)
        # action_idx = scores[self.action_set].argmax()env
        # action_idx = (self.action_set==action).nonzero().item()
        self.action_idx = torch.LongTensor(np.array([action_idx],dtype=np.int32))
        self.reward = torch.FloatTensor(np.expand_dims(reward, axis=-1))
        self.done = torch.FloatTensor(np.expand_dims(int(done), axis=-1))

        if isinstance(next_state,list):
            self.tree = True
            next_state_l,next_state_r = next_state[0],next_state[1]
            if next_state_l is not None:
                self.constraint_features_l = next_state_l.constraint_features
                self.edge_index_l = next_state_l.edge_index
                self.edge_attr_l = next_state_l.edge_attr
                self.variable_features_l = next_state_l.variable_features
                self.action_set_l = next_state_l.action_set
                self.action_set_l_size = next_state_l.action_set_size
                self.node_id_l = next_state_l.node_id
                self.num_nodes_l = next_state_l.num_nodes
                # assert self.edge_index_l.max()<self.variable_features_l.shape[0]
            else:
                print('Error in next state')

            if next_state_r is not None:
                self.constraint_features_r = next_state_r.constraint_features
                self.edge_index_r = next_state_r.edge_index
                self.edge_attr_r = next_state_r.edge_attr
                self.variable_features_r = next_state_r.variable_features
                self.action_set_r = next_state_r.action_set
                self.action_set_r_size = next_state_r.action_set_size
                self.node_id_r = next_state_r.node_id
                self.num_nodes_r = next_state_r.num_nodes
                # assert self.edge_index_r.max()<self.variable_features_r.shape[0]

            else:
                self.constraint_features_r = next_state_l.constraint_features
                self.edge_index_r = next_state_l.edge_index
                self.edge_attr_r = next_state_l.edge_attr
                self.variable_features_r = next_state_l.variable_features
                self.action_set_r = next_state_l.action_set
                self.action_set_r_size = next_state_l.action_set_size
                self.node_id_r = next_state_l.node_id
                self.num_nodes_r = next_state_l.num_nodes
        else:
            self.tree = False
            self.constraint_features_n = next_state.constraint_features
            self.edge_index_n = next_state.edge_index
            self.edge_attr_n = next_state.edge_attr
            self.variable_features_n = next_state.variable_features
            self.action_set_n = next_state.action_set
            self.action_set_n_size = next_state.action_set_size
            self.node_id_n = next_state.node_id
            self.num_nodes_n = next_state.num_nodes
        

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'edge_index_n':
            return torch.tensor([[self.constraint_features_n.size(0)], [self.variable_features_n.size(0)]])
        elif key == 'edge_index_l':
            return torch.tensor([[self.constraint_features_l.size(0)], [self.variable_features_l.size(0)]])
        elif key == 'edge_index_r':
            return torch.tensor([[self.constraint_features_r.size(0)], [self.variable_features_r.size(0)]])
        elif key == 'action_set':
            return self.variable_features.size(0)
        elif key == 'action_set_n':
            return self.variable_features_n.size(0)
        elif key == 'action_set_l':
            return self.variable_features_l.size(0)
        elif key == 'action_set_r':
            return self.variable_features_r.size(0)
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
    stats = []
    for sample_file in data_files:
        with gzip.open(sample_file, 'rb') as f:
            sample = pickle.load(f)
        if len(sample['data'])==7:
            state, action, action_idx, _, reward, done, next_state = sample['data']
        else:
            state, action, action_idx,  reward, done,  next_state = sample['data']
        if isinstance(next_state,list):
            if next_state[0] is None and next_state[1] is None:
                continue
        # if done:
        #     print(reward)
        fulltransition = FullTransition(state, action, action_idx, reward, done, next_state)
        transitions.append(fulltransition)
    random.shuffle(transitions)
    return transitions, stats

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
        _,action_set,_,_,_ = env.reset()
        iter_count = 0
        rewards = 0
        while True:
            action_idx = policy.get_action(env.state, eval=True)
            action = action_set[action_idx]
            observation, action_set, reward, done, info = env.step(action)
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
            agent.save( wandb=wandb ,stat=stat)
    return wandb_data

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class RetroBranching:
    def __init__(self, 
                 retro_trajectory_construction='max_leaf_lp_gain',
                 use_retro_trajectories=True,
                 only_use_leaves_closed_by_brancher_as_terminal_nodes=True,
                 debug_mode=False):
        '''
        Waits until end of episode to calculate rewards for each step, then retrospectively
        goes back through each step in the episode and calculates reward for that step.
        I.e. reward returned will be None until the end of the episode, at which
        point a dict mapping episode_step_idx for optimal path nodes to reward will be returned.
        
        Args:
            retro_trajectory_construction ('random', 'deepest', 'shortest', 'max_lp_gain', 'min_lp_gain', 'max_leaf_lp_gain', 
                'reverse_visitation_order', 'visitation_order'): Which policy to use when choosing a leaf node as the final 
                node to construct a retrospective trajectory.
            use_retro_trajectories (bool): If False, will return original dict mapping before forming retrospective episodes.
            only_use_leaves_closed_by_brancher_as_terminal_nodes (bool): If True, when constructing retrospective trajectories,
                will only consider leaves which were fathomed by the brancher as the last transition for a retrospective trajectory.
        '''
        self.retro_trajectory_construction = retro_trajectory_construction
        self.use_retro_trajectories = use_retro_trajectories
        self.only_use_leaves_closed_by_brancher_as_terminal_nodes = only_use_leaves_closed_by_brancher_as_terminal_nodes
        self.debug_mode = debug_mode

    def before_reset(self, model):
        self.started = False
        
    def get_path_node_scores(self, tree, path):
        # use original node score as score for each node
        return [tree.nodes[node]['score'] for node in path]

    def conv_path_to_step_idx_reward_map(self, path):        
        # register which nodes have been directly included in the sub-tree
        for node in path:
            self.nodes_added.add(node)
        
        # get rewards at each step in sub-tree episode
        path_node_rewards = self.get_path_node_scores(self.tree.tree, path)

        # get episode step indices at which each node in sub-tree was visited
        path_to_step_idx = {node: self.visited_nodes_to_step_idx[node] for node in path}

        # map each path node episode step idx to its corresponding reward
        step_idx_to_reward = {step_idx: r for step_idx, r in zip(list(path_to_step_idx.values()), path_node_rewards)}
        
        return step_idx_to_reward


    def _select_path_in_subtree(self, subtree):
        for root_node in subtree.nodes:
            if subtree.in_degree(root_node) == 0:
                # node is root
                break

        # use a construction method to select a sub-tree episode path through the sub-tree
        if self.retro_trajectory_construction == 'max_lp_gain' or self.retro_trajectory_construction == 'min_lp_gain':
            # iteratively decide next node in path at each step
            curr_node, path = root_node, [root_node]
            while True:
                # get potential next node(s)
                children = [child for child in subtree.successors(curr_node)]
                if len(children) == 0:
                    # curr node is final leaf node, path complete
                    break
                else:
                    # select next node
                    if self.retro_trajectory_construction == 'max_lp_gain':
                        idx = np.argmax([subtree.nodes[child]['lower_bound'] for child in children])
                    elif self.retro_trajectory_construction == 'min_lp_gain':
                        idx = np.argmin([subtree.nodes[child]['lower_bound'] for child in children])
                    else:
                        raise Exception(f'Unrecognised retro_trajectory_construction {self.retro_trajectory_construction}')
                    curr_node = children[idx]
                    path.append(curr_node)
        else:
            # first get leaf nodes and then use construction method to select leaf target for shortest path
            if self.only_use_leaves_closed_by_brancher_as_terminal_nodes:
                leaf_nodes = [node for node in subtree.nodes() if (subtree.out_degree(node) == 0 and node in self.tree.tree.graph['fathomed_node_ids'])]
            else:
                leaf_nodes = [node for node in subtree.nodes() if subtree.out_degree(node) == 0]
            
            if len(leaf_nodes) == 0:
                # could not find any valid path through sub-tree
                return []

            if self.retro_trajectory_construction == 'random':
                # randomly choose leaf node as final node
                final_node = leaf_nodes[random.choice(range(len(leaf_nodes)))]
            elif self.retro_trajectory_construction == 'deepest':
                # choose leaf node which would lead to deepest subtree as final node
                depths = [len(shortest_path(subtree, source=root_node, target=leaf_node)) for leaf_node in leaf_nodes]
                final_node = leaf_nodes[depths.index(max(depths))]
            elif self.retro_trajectory_construction == 'shortest':
                # choose leaf node which would lead to shortest subtree as final node
                depths = [len(shortest_path(subtree, source=root_node, target=leaf_node)) for leaf_node in leaf_nodes]
                final_node = leaf_nodes[depths.index(min(depths))]
            elif self.retro_trajectory_construction == 'max_leaf_lp_gain':
                # choose leaf node which has greatest LP gain as final node
                lp_gains = [subtree.nodes[leaf_node]['lower_bound'] for leaf_node in leaf_nodes]
                final_node = leaf_nodes[lp_gains.index(max(lp_gains))]
            elif self.retro_trajectory_construction == 'reverse_visitation_order':
                step_node_visited = [self.tree.tree.nodes[leaf_node]['step_visited'] for leaf_node in leaf_nodes]
                final_node = leaf_nodes[step_node_visited.index(max(step_node_visited))]
            elif self.retro_trajectory_construction == 'visitation_order':
                step_node_visited = [self.tree.tree.nodes[leaf_node]['step_visited'] for leaf_node in leaf_nodes]
                final_node = leaf_nodes[step_node_visited.index(min(step_node_visited))]
            else:
                raise Exception(f'Unrecognised retro_trajectory_construction {self.retro_trajectory_construction}')
            path = shortest_path(self.tree.tree, source=root_node, target=final_node)

        return path
    def extract(self, model, done):
        if not self.started:
            self.started = True
            self.tree = SearchTree(model)
            return None

        if not done:
            # update B&B tree
            self.tree.update_tree(model)
            return None
        
        else:
            # instance finished, retrospectively create subtree episode paths
            self.tree.update_tree(model)

            if self.tree.tree.graph['root_node'] is None:
                # instance was pre-solved
                return [{0: 0}]

            # collect sub-tree episodes
            subtrees_step_idx_to_reward = []

            # keep track of which nodes have been added to a sub-tree
            self.nodes_added = set()
            
            if self.debug_mode:
                print('\nB&B tree:')
                print(f'All nodes saved: {self.tree.tree.nodes()}')
                print(f'Visited nodes: {self.tree.tree.graph["visited_node_ids"]}')
                self.tree.render()

            # remove nodes which were never visited by the brancher and therefore do not have a score or next state
            nodes = [node for node in self.tree.tree.nodes]
            for node in nodes:
                if 'step_visited' not in self.tree.tree.nodes[node]:
                    self.tree.tree.remove_node(node)
                    if self.debug_mode:
                        print(f'Removing node {node} since was never visited by brancher.')
                    if node in self.tree.tree.graph['visited_node_ids']:
                        # hack: SCIP sometimes returns large int rather than None node_id when episode finished
                        # since never visited this node (since no score assigned), do not count this node as having been visited when calculating paths below
                        if self.debug_mode:
                            print(f'Removing node {node} from visied IDs since was never actually visited by brancher.')
                        self.tree.tree.graph['visited_node_ids'].remove(node)                    
                        
            # set node scores (transition rewards)
            for node in self.tree.tree.nodes:
                if node in self.tree.tree.graph['fathomed_node_ids']:
                    self.tree.tree.nodes[node]['score'] = 0
                else:
                    self.tree.tree.nodes[node]['score'] = -1

            # map which nodes were visited at which step in episode
            self.visited_nodes_to_step_idx = {node: idx for idx, node in enumerate(self.tree.tree.graph['visited_node_ids'])}

            if not self.use_retro_trajectories:
                # do not use any sub-tree episodes, just return whole B&B tree episode
                step_idx_to_reward = {}
                for node, step_idx in self.visited_nodes_to_step_idx.items():
                    step_idx_to_reward[step_idx] = self.tree.tree.nodes[node]['score']
                subtrees_step_idx_to_reward.append(step_idx_to_reward)
                return subtrees_step_idx_to_reward 
            
            root_node = list(self.tree.tree.graph['root_node'].keys())[0]
            # create sub-tree episodes from remaining B&B nodes visited by agent
            while True:
                # create depth first search sub-trees from nodes still leftover
                nx_subtrees = []
                
                # construct sub-trees containing prospective sub-tree episode(s) from remaining nodes
                if len(self.nodes_added) > 0:
                    for node in self.nodes_added:
                        children = [child for child in self.tree.tree.successors(node)]
                        for child in children:
                            if child not in self.nodes_added:
                                nx_subtrees.append(dfs_tree(self.tree.tree, child))
                else:
                    # not yet added any nodes to a sub-tree, whole B&B tree is first 'sub-tree'
                    nx_subtrees.append(dfs_tree(self.tree.tree, root_node))
                            
                for i, subtree in enumerate(nx_subtrees):
                    # init node attributes for nodes in subtree (since these are not transferred into new subtree by networkx)
                    for node in subtree.nodes:
                        subtree.nodes[node]['score'] = self.tree.tree.nodes[node]['score']
                        subtree.nodes[node]['lower_bound'] = self.tree.tree.nodes[node]['lower_bound']

                    # choose episode path through sub-tree
                    path = self._select_path_in_subtree(subtree)
                    
                    if len(path) > 0:
                        # gather rewards in sub-tree
                        subtree_step_idx_to_reward = self.conv_path_to_step_idx_reward_map(path)
                        if subtree_step_idx_to_reward is not None:
                            subtrees_step_idx_to_reward.append(subtree_step_idx_to_reward)
                        else:
                            # subtree was not deep enough to be added
                            pass
                    else:
                        # cannot establish valid path through sub-tree, do not consider nodes in this sub-tree again
                        for node in subtree.nodes():
                            self.nodes_added.add(node)

                if len(nx_subtrees) == 0:
                    # all sub-trees added
                    break
        if self.debug_mode:
            print(f'visited_nodes_to_step_idx: {self.visited_nodes_to_step_idx}')
            step_idx_to_visited_nodes = {val: key for key, val in self.visited_nodes_to_step_idx.items()}
            print(f'step_idx_to_visited_nodes: {step_idx_to_visited_nodes}')
            for i, ep in enumerate(subtrees_step_idx_to_reward):
                print(f'>>> sub-tree episode {i+1}: {ep}')
                ep_path = [step_idx_to_visited_nodes[idx] for idx in ep.keys()]
                print(f'path: {ep_path}')
                ep_dual_bounds = [self.tree.tree.nodes[node]['lower_bound'] for node in ep_path]
                print(f'ep_dual_bounds: {ep_dual_bounds}')
        
        if len(subtrees_step_idx_to_reward) == 0:
            # solved at root so path length < min path length so was never added to subtrees
            return [{0: 0}]
        else:
            return subtrees_step_idx_to_reward