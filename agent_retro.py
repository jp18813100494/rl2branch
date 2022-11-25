import ecole
import threading
import queue
import utilities
import random
import numpy as np
from collections import namedtuple
from utilities import RetroBranching

class AgentPool():
    """
    Class holding the reference to the agents and the policy sampler.
    Puts jobs in the queue through job sponsors.
    """
    def __init__(self, brain, n_agents, time_limit, mode):
        self.jobs_queue = queue.Queue()
        self.policy_queries_queue = queue.Queue()
        self.policy_sampler = PolicySampler("Policy Sampler", brain, self.policy_queries_queue)
        self.agents = [Agent(f"Agent {i}", time_limit, self.jobs_queue, self.policy_queries_queue, mode) for i in range(n_agents)]

    def start(self):
        self.policy_sampler.start()
        for agent in self.agents:
            agent.start()

    def close(self):
        # order the episode sampling agents to stop
        for _ in self.agents:
            self.jobs_queue.put(None)
        self.jobs_queue.join()
        # order the policy sampler to stop
        self.policy_queries_queue.put(None)
        self.policy_queries_queue.join()

    def start_job(self, instances, sample_rate, greedy=False, block_policy=False):
        """
        Starts a job.
        A job is a set of tasks. A task consists of an instance that needs to be solved and instructions
        to do so (sample rate, greediness).
        The job queue is loaded with references to the job sponsor, which is in itself a queue specific
        to a job. It is the job sponsor who holds the lists of tasks. The role of the job sponsor is to
        keep track of which tasks have been completed.
        """
        job_sponsor = queue.Queue()
        samples = []
        stats = []

        policy_access = threading.Event()
        if not block_policy:
            policy_access.set()

        for instance in instances:
            task = {'instance': instance, 'sample_rate': sample_rate, 'greedy': greedy,
                    'samples': samples, 'stats': stats, 'policy_access': policy_access}
            job_sponsor.put(task)
            self.jobs_queue.put(job_sponsor)
        samples = random.shuffle(samples)
        ret = (samples, stats, job_sponsor)
        if block_policy:
            ret = (*ret, policy_access)

        return ret

    def wait_completion(self):
        # wait for all running episodes to finish
        self.jobs_queue.join()


class PolicySampler(threading.Thread):
    """
    Gathers policy sampling requests from the agents, and process them in a batch.
    """
    def __init__(self, name, brain, requests_queue):
        super().__init__(name=name)
        self.brain = brain
        self.requests_queue = requests_queue

    def run(self):
        stop_order_received = False
        while True:
            requests = []
            request = self.requests_queue.get()
            while True:
                # check for a stopping order
                if request is None:
                    self.requests_queue.task_done()
                    stop_order_received = True
                    break
                # add request to the batch
                requests.append(request)
                # keep collecting more requests if available, without waiting
                try:
                    request = self.requests_queue.get(block=False)
                except queue.Empty:
                    break

            states = [r['state'] for r in requests]
            greedys = [r['greedy'] for r in requests]
            receivers = [r['receiver'] for r in requests]

            # process all requests in a batch
            action_idxs = self.brain.sample_action_idx(states, greedys)
            for action_idx, receiver in zip(action_idxs, receivers):
                receiver.put(action_idx)
                self.requests_queue.task_done()

            if stop_order_received:
                break


class Agent(threading.Thread):
    """
    Agent class. Receives tasks from the job sponsor, runs them and samples transitions if
    requested.
    """
    def __init__(self, name, time_limit, jobs_queue, policy_queries_queue, mode):
        super().__init__(name=name)
        self.jobs_queue = jobs_queue
        self.policy_queries_queue = policy_queries_queue
        self.policy_answers_queue = queue.Queue()
        self.mode = mode

        # Setup Ecole environment
        scip_params={'separating/maxrounds': 0,
                     'presolving/maxrestarts': 0,
                     'limits/time': time_limit,
                     'timing/clocktype': 2}
        observation_function=(
            ecole.observation.FocusNode(),
            ecole.observation.NodeBipartite()
            )
        # reward_function = ecole.reward.NNodes().cumsum()
        reward_function= {
            "cum_nnodes": ecole.reward.NNodes().cumsum(),
            "cur_nnodes": ecole.reward.NNodes()
        } 
        
        information_function={
            'nnodes': ecole.reward.NNodes().cumsum(),
            'lpiters': ecole.reward.LpIterations().cumsum(),
            'time': ecole.reward.SolvingTime().cumsum()
        }

        if mode == 'tmdp+ObjLim':
            self.env = ObjLimBranchingEnv(scip_params=scip_params,
                                          pseudo_candidates=False,
                                          observation_function=observation_function,
                                          reward_function=reward_function,
                                          information_function=information_function)
        elif mode == 'tmdp+DFS':
            self.env = DFSBranchingEnv(scip_params=scip_params,
                                       pseudo_candidates=False,
                                       observation_function=observation_function,
                                       reward_function=reward_function,
                                       information_function=information_function)
        elif mode == 'mdp':
            self.env = MDPBranchingEnv(scip_params=scip_params,
                                       pseudo_candidates=False,
                                       observation_function=observation_function,
                                       reward_function=reward_function,
                                       information_function=information_function)
        elif mode == 'retro' or mode == 'retro2':
            self.env = ecole.environment.Branching(scip_params=scip_params,
                                       observation_function=observation_function,
                                       reward_function=reward_function,
                                       information_function=information_function)
        
        else:
            raise NotImplementedError

    def run(self):
        while True:
            job_sponsor = self.jobs_queue.get()

            # check for a stopping order
            if job_sponsor is None:
                self.jobs_queue.task_done()
                break

            # Get task from job sponsor
            task = job_sponsor.get()
            instance = task['instance']
            sample_rate = task['sample_rate']
            greedy = task['greedy']   # should actions be chosen greedily w.r.t. the policy?
            training = not greedy
            samples = task['samples']
            stats = task['stats']
            policy_access = task['policy_access']
            seed = instance['seed']
            instance = instance['path']

            transitions = []
            # initialise custom reward object
            custom_reward = RetroBranching(debug_mode=False)
            # initialise an instance which is not pre-solved by SCIP
            self.env.seed(seed)
            custom_reward.before_reset(instance)
            obs, action_set, reward, done, info = self.env.reset(instance)
            _custom_reward = custom_reward.extract(self.env.model, done)

            # # Run episode
            policy_access.wait()
            iter_count = 0
            prev_obs = obs
            if not done:
                focus_node_obs,node_observation = obs
                pre_state = utilities.extract_state(node_observation, action_set, focus_node_obs.number)
            original_transitions = []
            while not done:
                 # send out policy queries
                self.policy_queries_queue.put({'state': pre_state, 'greedy': greedy, 'receiver': self.policy_answers_queue})
                action_idx = self.policy_answers_queue.get()
                action = action_set[action_idx]
                obs, action_set, reward, done, info = self.env.step(action)
                _custom_reward = custom_reward.extract(self.env.model, done)

                if done:
                    obs = prev_obs
                    state = pre_state
                else:
                    focus_node_obs,node_observation = obs
                    state = utilities.extract_state(node_observation, action_set, focus_node_obs.number)
                if sample_rate > 0:
                    original_transitions.append({'obs': prev_obs,
                                            'state': pre_state,
                                            'action': action,
                                            'action_idx': action_idx,
                                            'reward': None,
                                            'done': done,
                                            'next_obs': obs,
                                            'next_state': state}) 
                iter_count += 1
                # update prev obs
                prev_obs = obs
                pre_state = state
                if (iter_count>50000) and training: done=True # avoid too large trees during training for stability

            if (iter_count>50000) and training: # avoid too large trees during training for stability
                job_sponsor.task_done()
                self.jobs_queue.task_done()
                continue

            # post-process the collected samples (credit assignment)
            if sample_rate > 0 and _custom_reward:
                for retro_traj in _custom_reward:
                    t = 0
                    path_nodes = list(retro_traj.keys())
                    for step_idx, reward in retro_traj.items():
                        transition = original_transitions[step_idx]
                        transition['reward'] = reward
                        if self.mode == 'retro':
                            if t < len(path_nodes)-1:
                                transition['next_state'] = original_transitions[path_nodes[t+1]]['state']
                            else:
                                transition['done'] = True
                        transition = utilities.FullTransition(transition['state'], transition['action'], transition['action_idx'], transition['reward'], transition['done'], transition['next_state'], 0)
                        transitions.append(transition)
                        t += 1

            # record episode samples and stats
            samples.extend(transitions)
            stats.append({'order': task, 'info': info})

            # tell both the agent pool and the original task sponsor that the task is done
            job_sponsor.task_done()
            self.jobs_queue.task_done()


class TreeRecorder:
    """
    Records the branch-and-bound tree from a custom brancher.

    Every node in SCIP has a unique node ID. We identify nodes and their corresponding
    attributes through the same ID system.
    Depth groups keep track of groups of nodes at the same depth. This data structure
    is used to speed up the computation of the subtree size.
    """
    def __init__(self):
        self.tree = {}
        self.depth_groups = []

    def record_branching_decision(self, focus_node, lp_cand=True):
        id = focus_node.number
        # Tree
        self.tree[id] = {'parent': focus_node.parent_number,
                         'lowerbound': focus_node.lowerbound,
                         'num_children': 2 if lp_cand else 3  }
        # Add to corresponding depth group
        if len(self.depth_groups) > focus_node.depth:
            self.depth_groups[focus_node.depth].append(id)
        else:
            self.depth_groups.append([id])

    def calculate_subtree_sizes(self):
        subtree_sizes = {id: 0 for id in self.tree.keys()}
        for group in self.depth_groups[::-1]:
            for id in group:
                parent_id = self.tree[id]['parent']
                subtree_sizes[id] += self.tree[id]['num_children']
                if parent_id >= 0: subtree_sizes[parent_id] += subtree_sizes[id]
        return subtree_sizes
    
    def calculate_subtree_sizes_next_state(self, datas, mode):
        subtree_sizes = {id: 0 for id in self.tree.keys()}
        for group in self.depth_groups[::-1]:
            for id in group:
                parent_id = self.tree[id]['parent']
                subtree_sizes[id] += self.tree[id]['num_children']
                if parent_id >= 0: 
                    subtree_sizes[parent_id] += subtree_sizes[id]
                    if datas[parent_id]['next_state'][0] == None:
                        datas[parent_id]['next_state'][0] = datas[id]['state']
                    else:
                        assert datas[parent_id]['next_state'][1]==None
                        datas[parent_id]['next_state'][1] = datas[id]['state']
        total_nnodes = datas.get(list(datas.keys())[-1])['cum_nnodes']
        for group in self.depth_groups:
            for id in group:     
                if mode in ['tmdp+ObjLim', 'tmdp+DFS']:
                    datas[id]['reward'] = datas[id]['returns'] = -subtree_sizes[id] - 1
                else:
                    assert mode == 'mdp'
                    datas[id]['reward'] = datas[id]['returns'] = datas[id]['cum_nnodes'] - total_nnodes

        for group in self.depth_groups[::-1]:
            for id in group:
                parent_id = self.tree[id]['parent']
                if parent_id>=0:
                    datas[parent_id]['reward'] -= datas[id]['returns']
        return datas


class DFSBranchingDynamics(ecole.dynamics.BranchingDynamics):
    """
    Custom branching environment that changes the node strategy to DFS when training.
    """
    def reset_dynamics(self, model, primal_bound, training, *args, **kwargs):
        pyscipopt_model = model.as_pyscipopt()
        if training:
            # Set the dfs node selector as the least important
            pyscipopt_model.setParam(f"nodeselection/dfs/stdpriority", 666666)
            pyscipopt_model.setParam(f"nodeselection/dfs/memsavepriority", 666666)
        else:
            # Set the dfs node selector as the most important
            pyscipopt_model.setParam(f"nodeselection/dfs/stdpriority", 0)
            pyscipopt_model.setParam(f"nodeselection/dfs/memsavepriority", 0)

        return super().reset_dynamics(model, *args, **kwargs)

class DFSBranchingEnv(ecole.environment.Environment):
    __Dynamics__ = DFSBranchingDynamics

class ObjLimBranchingDynamics(ecole.dynamics.BranchingDynamics):
    """
    Custom branching environment that allows the user to set an initial primal bound.
    """
    def reset_dynamics(self, model, primal_bound, training, *args, **kwargs):
        pyscipopt_model = model.as_pyscipopt()
        if primal_bound is not None:
            pyscipopt_model.setObjlimit(primal_bound)

        return super().reset_dynamics(model, *args, **kwargs)

class ObjLimBranchingEnv(ecole.environment.Environment):
    __Dynamics__ = ObjLimBranchingDynamics

class MDPBranchingDynamics(ecole.dynamics.BranchingDynamics):
    """
    Regular branching environment that allows extra input parameters, but does
    not use them.
    """
    def reset_dynamics(self, model, primal_bound, training, *args, **kwargs):
        return super().reset_dynamics(model, *args, **kwargs)

class MDPBranchingEnv(ecole.environment.Environment):
    __Dynamics__ = MDPBranchingDynamics
