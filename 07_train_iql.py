# import gym
# import d4rl
import pathlib
import os
import os.path as osp
import sys
import datetime
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
import torch_geometric
import wandb
import argparse
import json
import utilities
# from utils import save, collect_random
import random
from iql.iql_agent import IQL,save
from utilities import pad_tensor, ExtendGraphDataset, Scheduler,BuildFullTransition
from envs.branch_env import branch_env
from scipy.stats.mstats import gmean

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="IQL", help="Run name, default: SAC")
    # parser.add_argument("--env", type=str, default="halfcheetah-medium-v2", help="Gym environment name, default: Pendulum-v0")
    # parser.add_argument("--max_epochs", type=int, default=1000, help="Number of episodes, default: 100")
    # parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default: 256")
    parser.add_argument("--valid_batch_size", type=int, default=128, help="Valid Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning_rate")
    parser.add_argument("--temperature", type=float, default=3, help="")
    parser.add_argument("--expectile", type=float, default=0.7, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--gamma", type=float, default=0.99, help="")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    parser.add_argument('-p','--problem',help='MILP instance type to process.',choices=['setcover', 'cauctions', 'ufacilities', 'indset', 'mknapsack'],default='setcover',)
    parser.add_argument('-s', '--seed',help='Random generator seed.',type=int,default=0)
    parser.add_argument('-g', '--gpu',help='CUDA GPU id (-1 for CPU).',type=int,default=0,)
    parser.add_argument("--max_epochs", type=int, default=1000, help="Number of max_epochs, default: 1000")
    parser.add_argument("--mode", type=str, default='mdp', help="Mode for branch env")
    parser.add_argument("--time_limit", type=int, default=4000, help="Timelimit of the solver")
    parser.add_argument('--wandb',help="Use wandb?",default=False,action="store_true")
    
    
    args = parser.parse_args()
    return args

def prep_dataloader(data_files, sol_sets, config,train=True):
    env =branch_env(data_files,sol_sets,config)
    # dataset = ExtendGraphDataset(data_files)
    samples = BuildFullTransition(data_files)
    if train:
        batch_size = config.batch_size
    else:
        batch_size = config.valid_batch_size
    dataloader  = torch_geometric.data.DataLoader(samples, batch_size=batch_size, shuffle=True)
    return dataloader, env

def evaluate(env, policy, eval_runs=5): 
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    stats = []
    for i in range(eval_runs):
        state = env.reset()

        rewards = 0
        while True:
            action = policy.get_action(state, eval=True)
            state, reward, done, info = env.step(action)
            rewards += reward
            stats.append({'task':env.instance,'info':info})
            if done:
                break
        reward_batch.append(rewards)
    return reward_batch, stats

def train(config):
    #TODO: Data_driven + Environment
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    token = '{}-{}-{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), config.run_name, config.mode)
    model_dir = osp.realpath(osp.join('results', token))
    config.model_dir = model_dir

     # data
    if config.problem == "setcover":
        maximization = False
        valid_path = "data/instances/setcover/valid_400r_750c_0.05d"
        train_path = "data/instances/setcover/train_400r_750c_0.05d"
    elif config.problem == "cauctions":
        maximization = True
        valid_path = "data/instances/cauctions/valid_100_500"
        train_path = "data/instances/cauctions/train_100_500"
    elif config.problem == "indset":
        maximization = True
        valid_path = "data/instances/indset/valid_500_4"
        train_path = "data/instances/indset/train_500_4"
    elif config.problem == "ufacilities":
        maximization = False
        valid_path = "data/instances/ufacilities/valid_35_35_5"
        train_path = "data/instances/ufacilities/train_35_35_5"
    elif config.problem == "mknapsack":
        maximization = True
        valid_path = "data/instances/mknapsack/valid_100_6"
        train_path = "data/instances/mknapsack/train_100_6"
    else:
        raise NotImplementedError
    config.maximization = maximization

    problem_folders = {
        'setcover': 'setcover/400r_750c_0.05d',
        'cauctions': 'cauctions/100_500',
        'ufacilities': 'ufacilities/35_35_5',
        'indset': 'indset/500_4',
        'mknapsack': 'mknapsack/100_6',
    }
    problem_folder = problem_folders[config.problem]
    running_dir = f"actor/{config.problem}/{config.seed}"
    os.makedirs(running_dir, exist_ok=True)

    ### PYTORCH SETUP ###
    if config.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{config.gpu}'
        device = f"cuda:0"

    sys.path.insert(0, os.path.abspath(f'./actor'))

    ### LOG ###
    logger = utilities.configure_logging()
    if config.wandb:
        wandb.init(project="rl2branch", name='IQL', config=config)

    train_files = [str(file) for file in (pathlib.Path(f'data/samples_orl')/problem_folder/'train').glob('sample_*.pkl')]
    valid_files = [str(file) for file in (pathlib.Path(f'data/samples_orl')/problem_folder/'valid').glob('sample_*.pkl')]
    logger.info(f"Training on {len(train_files)} training instances and {len(valid_files)} validation instances")
    # collect the pre-computed optimal solutions for the training instances
    with open(f"{train_path}/instance_solutions.json", "r") as f:
        train_sols = json.load(f)
    with open(f"{valid_path}/instance_solutions.json", "r") as f:
        valid_sols = json.load(f)
    # dataloader, env = prep_dataloader(train_files,train_sols,config,train=True)
    _, env = prep_dataloader(valid_files,valid_sols,config,train=False)

    batches = 0
    average10 = deque(maxlen=10)
    
    agent = IQL(state_size=env.observation_space.shape[0],
                action_size=env.action_space.shape[0],
                learning_rate=config.lr,
                hidden_size=config.hidden_size,
                tau=config.tau,
                gamma=config.gamma,
                temperature=config.temperature,
                expectile=config.expectile,
                seed = config.seed,
                device=device)
    rng = np.random.RandomState(config.seed)
    wandb.watch(agent, log="gradients", log_freq=10)
    eval_reward,_ = evaluate(env, agent)
    logger.info(f"Test Reward: {eval_reward}, Episode: 0, Batches: {batches}")
    for epoch in range(1, config.max_epochs+1):
        logger.info(f'** Epoch {epoch}')
        wandb_data = {}
        epoch_train_files = rng.choice(train_files, int(np.floor(10000/config.batch_size))*config.batch_size, replace=True)
        train_data = ExtendGraphDataset(epoch_train_files)
        train_loader = torch_geometric.data.DataLoader(train_data, config.batch_size, shuffle=True)
        for batch_idx, batch in enumerate(train_loader):
            batch.to(config.device)
            #TODO: revise this part
            states, actions, rewards, next_states, dones = batch
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            policy_loss, critic1_loss, critic2_loss, value_loss = agent.learn((states, actions, rewards, next_states, dones))
            batches += 1

        if epoch % config.eval_every == 0:
            eval_reward, v_stats = evaluate(env, agent)
            v_nnodess = [s['info']['nnodes'] for s in v_stats]
            v_lpiterss = [s['info']['lpiters'] for s in v_stats]
            v_times = [s['info']['time'] for s in v_stats]

            wandb_data.update({
                'valid_reward':np.mena(eval_reward),
                'valid_nnodes_g': gmean(np.asarray(v_nnodess) + 1) - 1,
                'valid_nnodes': np.mean(v_nnodess),
                'valid_nnodes_max': np.amax(v_nnodess),
                'valid_nnodes_min': np.amin(v_nnodess),
                'valid_time': np.mean(v_times),
                'valid_lpiters': np.mean(v_lpiterss),
            })
            if epoch == 0:
                v_nnodes_0 = wandb_data['valid_nnodes'] if wandb_data['valid_nnodes'] != 0 else 1
                v_nnodes_g_0 = wandb_data['valid_nnodes_g'] if wandb_data['valid_nnodes_g']!= 0 else 1
            wandb_data.update({
                'valid_nnodes_norm': wandb_data['valid_nnodes'] / v_nnodes_0,
                'valid_nnodes_g_norm': wandb_data['valid_nnodes_g'] / v_nnodes_g_0,
            })

            if wandb_data['valid_nnodes_g'] < best_tree_size:
                best_tree_size = wandb_data['valid_nnodes_g']
                logger.info('Best parameters so far (1-shifted geometric mean), saving model.')
                if config.wandb:
                    save(config, model_dir,model=agent, wandb=wandb)
            average10.append(eval_reward)
            
            logger.info(f"Episode: {epoch} | Reward: {eval_reward} | Polciy Loss: {policy_loss} | Batches: {batches}")
        
        wandb.update({
                    "Valid_reward10": np.mean(average10),
                    "Policy Loss": policy_loss,
                    "Value Loss": value_loss,
                    "Critic 1 Loss": critic1_loss,
                    "Critic 2 Loss": critic2_loss,
                    "Batches": batches,
                    "Episode": epoch})

        # Send the stats to wandb
        if config.wandb:
            wandb.log(wandb_data, step = epoch)

        if epoch % config.save_every == 0 and config.wandb:
            save(config,  model_dir, model=agent, wandb=wandb, ep=epoch)

if __name__ == "__main__":
    config = get_config()
    train(config)
