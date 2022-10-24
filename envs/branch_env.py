import ecole
import numpy as np

import utilities
from agent import TreeRecorder
from gym import spaces

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


class base_env(object):
    def __init__(self, mode, time_limit,query_expert_prob):
        self.mode = mode
        self.time_limit = time_limit
        self.query_expert_prob = query_expert_prob

        # Setup Ecole environment
        self.scip_params={'separating/maxrounds': 0,
                     'presolving/maxrestarts': 0,
                     'limits/time': time_limit,
                     'timing/clocktype': 2}
        # self.observation_function=(
        #     ecole.observation.FocusNode(),
        #     ecole.observation.NodeBipartite()
        #     )
        self.observation_function = { "scores": ExploreThenStrongBranch(expert_probability=query_expert_prob),
                                "focus_node":ecole.observation.FocusNode(),
                                 "node_observation": ecole.observation.NodeBipartite() }
        self.reward_function=ecole.reward.NNodes()
        self.information_function={
            'nnodes': ecole.reward.NNodes().cumsum(),
            'lpiters': ecole.reward.LpIterations().cumsum(),
            'time': ecole.reward.SolvingTime().cumsum()
        }

        if mode == 'tmdp+ObjLim':
            self.env = ObjLimBranchingEnv(scip_params=self.scip_params,
                                          pseudo_candidates=False,
                                          observation_function=self.observation_function,
                                          reward_function=self.reward_function,
                                          information_function=self.information_function)
        elif mode == 'tmdp+DFS':
            self.env = DFSBranchingEnv(scip_params=self.scip_params,
                                       pseudo_candidates=False,
                                       observation_function=self.observation_function,
                                       reward_function=self.reward_function,
                                       information_function=self.information_function)
        elif mode == 'mdp':
            self.env = MDPBranchingEnv(scip_params=self.scip_params,
                                       pseudo_candidates=False,
                                       observation_function=self.observation_function,
                                       reward_function=self.reward_function,
                                       information_function=self.information_function)
        else:
            raise NotImplementedError


class branch_env(base_env):
    def __init__(self, instance_set, sol_sets, config, query_expert_prob=0.5):
        super().__init__(config['mode'],config['time_limit'],query_expert_prob)
        self.instance_set = instance_set
        self.sol_sets = sol_sets
        # self.seed = config['seed']
        self.train_size = len(instance_set)

        self.sample_rate = 0
        self.eps = -0.1 if config['maximization'] else 0.1
        #shuffle，seed
        self.instance_ind = 0
        self.epoch_shuffle_inds = np.arange(self.train_size)
        lower = np.array([0]*1,dtype=np.float32)
        upper = np.array([1]*1,dtype=np.float32)
        self.observation_space = spaces.Box(low = lower, high = upper, shape = (1,), dtype=np.float32)
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (1,), dtype=np.float32)

    def seed(self, seed_num):
        self.seed_num = seed_num
        self.rng = np.random.RandomState(self.seed_num)

    def sample_instance(self):
        if self.instance_ind % self.train_size == 0:
            np.random.shuffle(self.epoch_shuffle_inds)
            self.instance_ind = 0
        instance_file = self.instance_set[self.epoch_shuffle_inds[self.instance_ind]]
        self.instance_ind += 1
        return instance_file

    def initialize(self,instance,training=False):
        # Run episode
        self.instance = instance
        sol = self.sol_sets[instance] if instance in self.sol_sets else None
        self.observation, self.action_set, self.reward, self.done, self.info = self.env.reset(instance = instance,
                                                                                            primal_bound=sol+self.eps,
                                                                                            training=training)
        if self.observation is None:
            return False
        scores, scores_are_expert = self.observation["scores"]
        self.focus_node_obs = self.observation["focus_node"]
        node_bipartite_obs = self.observation["node_observation"]
        # self.focus_node_obs, node_bipartite_obs = self.observation
        self.state = utilities.extract_state(node_bipartite_obs, self.action_set, self.focus_node_obs.number)
        self.tree_recorder = TreeRecorder()
        self.transitions = []
        return True

    def reset(self, instance_file=None,sample_rate=0,training=False):
        self.sample_rate = sample_rate
        
        if instance_file is None:
            while True:
                instance_file = self.sample_instance()
                if (self.initialize(instance_file,training)):
                    break
            self.load_success = True
        else:
            self.load_success = self.initialize(instance_file,training)
        return self.observation, self.action_set, self.reward, self.done, self.info

    def step(self, action):
        # action = self.action_set[action_idx]
        # collect transition samples if requested
        if self.sample_rate > 0:
            self.tree_recorder.record_branching_decision(self.focus_node_obs)
            keep_sample = self.rng.rand() < self.sample_rate
            if keep_sample:
                transition = utilities.Transition(self.state, action, self.reward)
                self.transitions.append(transition)

        self.observation, self.action_set, self.reward, self.done, self.info = self.env.step(action)

        if self.done or self.observation == None:
            # post-process the collected samples (credit assignment)
            if self.sample_rate > 0:
                if self.mode in ['tmdp+ObjLim', 'tmdp+DFS']:
                    subtree_sizes = self.tree_recorder.calculate_subtree_sizes()
                    for transition in self.transitions:
                        transition.returns = -subtree_sizes[transition.node_id] - 1
                else:
                    assert self.mode == 'mdp'
                    for transition in self.transitions:
                        transition.returns = transition.cum_nnodes - self.reward
        else:
            # self.focus_node_obs, node_bipartite_obs = self.observation
            scores, scores_are_expert = self.observation["scores"]
            self.focus_node_obs = self.observation["focus_node"]
            node_bipartite_obs = self.observation["node_observation"]
            self.state = utilities.extract_state(node_bipartite_obs, self.action_set, self.focus_node_obs.number)
        return self.observation,self.action_set, self.reward, self.done, self.info

    

    


        
