import pathlib
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os.path as osp
import sys
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import wandb
import argparse
import json
import utilities
import random
import glob
from collections import deque
from iql.iql_agent import IQL,save
from utilities import Scheduler,BuildFullTransition
from envs.branch_env import branch_env
from scipy.stats.mstats import gmean

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--config', type=str, default='config/setcover/config_iql_mdp.json', help='path to yaml config file')
    parser.add_argument("--algo_name", type=str, default="IQL", help="Run name, default: SAC")
    parser.add_argument("--hidden_size", type=int, default=64, help="")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning_rate")
    parser.add_argument("--temperature", type=float, default=3, help="")
    parser.add_argument("--expectile", type=float, default=0.7, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--gamma", type=float, default=0.99, help="")
    parser.add_argument("--hard_update_every", type=int, default=10, help="")
    parser.add_argument("--clip_grad_param", type=int, default=100, help="")

    parser.add_argument("--max_epochs", type=int, default=200, help="Number of max_epochs, default: 1000")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default: 256")
    parser.add_argument("--valid_batch_size", type=int, default=128, help="Valid Batch size, default: 256")
    parser.add_argument("--num_valid_instances", type=int, default=1000, help="Number of valid instances for branch_env")
    
    parser.add_argument('--problem',help='MILP instance type to process.',choices=['setcover', 'cauctions', 'ufacilities', 'indset', 'mknapsack'],default='setcover',)
    parser.add_argument("--mode", type=str, default='mdp', help="Mode for branch env")
    parser.add_argument("--time_limit", type=int, default=4000, help="Timelimit of the solver")
    parser.add_argument("--sample_rate", type=float, default=0.05, help="")

    parser.add_argument('--seed',help='Random generator seed.',type=int,default=0)
    parser.add_argument('--gpu',help='CUDA GPU id (-1 for CPU).',type=int,default=0,)
    parser.add_argument('--wandb',help="Use wandb?",default=False,action="store_true")
    args = parser.parse_args()

    # override args with the user args file if provided
    args.config = osp.realpath(args.config)
    with open(args.config, 'r') as f:
        config = json.load(f)

    # override args with command-line arguments if provided
    args_config = {key: getattr(args, key) for key in config.keys() & vars(args).keys()}
    config.update(args_config)

    config['hard_update_every'] = int(np.floor(10000/config['batch_size']))
    ### Device SETUP ###
    if config['gpu'] == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        config['device'] = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{config['gpu']}"
        config['device'] = f"cuda:0"

    # data path for different problems
    if config['problem'] == "setcover":
        maximization = False
        valid_path = "data/instances/setcover/valid_400r_750c_0.05d"
        train_path = "data/instances/setcover/train_400r_750c_0.05d"
    elif config['problem'] == "cauctions":
        maximization = True
        valid_path = "data/instances/cauctions/valid_100_500"
        train_path = "data/instances/cauctions/train_100_500"
    elif config['problem'] == "indset":
        maximization = True
        valid_path = "data/instances/indset/valid_500_4"
        train_path = "data/instances/indset/train_500_4"
    elif config['problem'] == "ufacilities":
        maximization = False
        valid_path = "data/instances/ufacilities/valid_35_35_5"
        train_path = "data/instances/ufacilities/train_35_35_5"
    elif config['problem'] == "mknapsack":
        maximization = True
        valid_path = "data/instances/mknapsack/valid_100_6"
        train_path = "data/instances/mknapsack/train_100_6"
    else:
        raise NotImplementedError
    config['maximization'] = maximization
        # model path
    cur_name = '{}-{}-{}'.format(config['algo_name'], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),  config['mode'])
    token = '{}/{}'.format(config['problem'], cur_name)
    model_dir = osp.realpath(osp.join('results', token))
    config['model_dir'] = model_dir
    config['cur_name'] = cur_name
    config['train_path'] = train_path
    config['valid_path'] = valid_path

    problem_folders = {
        'setcover': 'setcover/400r_750c_0.05d',
        'cauctions': 'cauctions/100_500',
        'ufacilities': 'ufacilities/35_35_5',
        'indset': 'indset/500_4',
        'mknapsack': 'mknapsack/100_6',
    }
    config['problem_folder'] = problem_folders[config['problem']]
    return config

def evaluate(env, policy, eval_runs=20): 
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
        stats.append({'task':env.instance,'info':info})
        reward_batch.append(rewards)
    return reward_batch, stats

def train(config):
    #TODO: Data_driven + Environment
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    sys.path.insert(0, os.path.abspath(f'./results'))

    ### LOG ###
    logger = utilities.configure_logging()
    if config['wandb']:
        wandb.init(project="rl2branch", name=config['cur_name'], config=config)

    # recover training data & validation instances
    train_files = [str(file) for file in (pathlib.Path(f'data/samples_orl')/config['problem_folder']/'train').glob('sample_*.pkl')]
    valid_path = config['valid_path']
    valid_instances = [f'{valid_path}/instance_{j+1}.lp' for j in range(config['num_valid_instances'])]
    logger.info(f"Training on {len(train_files)} training instances and {len(valid_instances)} validation instances")
    # collect the pre-computed optimal solutions for the training instances
    with open(f"{config['train_path']}/instance_solutions.json", "r") as f:
        train_sols = json.load(f)
    with open(f"{config['valid_path']}/instance_solutions.json", "r") as f:
        valid_sols = json.load(f)

    env = branch_env(valid_instances,valid_sols,config)

    batches = 0
    average10 = deque(maxlen=10)
    
    agent = IQL(state_size=env.observation_space.shape[0],
                action_size=env.action_space.shape[0],
                learning_rate=config['lr'],
                hidden_size=config['hidden_size'],
                tau=config['tau'],
                gamma=config['gamma'],
                temperature=config['temperature'],
                expectile=config['expectile'],
                hard_update_every=config['hard_update_every'],
                clip_grad_param=config['clip_grad_param'],
                seed = config['seed'],
                device=config['device'])
    scheduler = Scheduler(agent.actor_optimizer, mode='min', patience=10, factor=0.2, verbose=True)
    rng = np.random.RandomState(config['seed'])
    eval_reward,_ = evaluate(env, agent)
    best_tree_size = np.inf
    logger.info(f"Test Reward: {eval_reward}, Episode: 0, Batches: {batches}")
    for epoch in range(0, config['max_epochs']+1):
        logger.info(f'** Epoch {epoch}')
        wandb_data = {}
        epoch_train_files = rng.choice(train_files, int(np.floor(10000/config['batch_size']))*config['batch_size'], replace=True)
        train_data = BuildFullTransition(epoch_train_files)
        train_loader = torch_geometric.data.DataLoader(train_data, config['batch_size'], shuffle=True)
        policy_losses, critic1_losses, critic2_losses, value_losses = [],[],[],[]
        for batch_idx, batch in enumerate(train_loader):
            batch.to(config['device'])
            policy_loss, critic1_loss, critic2_loss, value_loss = agent.learn(batch)
            policy_losses.append(policy_loss)
            critic1_losses.append(critic1_loss)
            critic2_losses.append(critic2_loss)
            value_losses.append(value_loss)
            batches += 1

        if epoch % config['eval_every'] == 0:
            eval_reward, v_stats = evaluate(env, agent)
            v_nnodess = [s['info']['nnodes'] for s in v_stats]
            v_lpiterss = [s['info']['lpiters'] for s in v_stats]
            v_times = [s['info']['time'] for s in v_stats]

            wandb_data.update({
                'valid_reward':np.mean(eval_reward),
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
                if config['wandb']:
                    save(config, config['model_dir'],model=agent, wandb=wandb)
            average10.append(eval_reward)
            logger.info(f"Episode: {epoch} | Reward: {eval_reward} | Polciy Loss: {np.mean(policy_losses)} | Batches: {batches}")
        
        wandb_data.update({
            "Valid_reward10": np.mean(average10),
            "Policy Loss": np.mean(policy_losses),
            "Value Loss": np.mean(value_losses),
            "Critic 1 Loss": np.mean(critic1_losses),
            "Critic 2 Loss": np.mean(critic2_losses),
            "Batches": batches,
            "Episode": epoch
        })

        # Send the stats to wandb
        if config['wandb']:
            wandb.log(wandb_data, step = epoch)

        if epoch % config['save_every'] == 0 and config['wandb']:
            save(config,  config['model_dir'], model=agent, wandb=wandb, ep=epoch)
        
        scheduler.step(np.mean(policy_losses))
        if config['wandb'] and scheduler.num_bad_epochs == 0:
            torch.save(agent.actor_local.state_dict(), pathlib.Path(config['model_dir'])/'iql_best_actor.pkl')
            logger.info(f"best model so far")
        elif scheduler.num_bad_epochs == 10:
            logger.info(f"  10 epochs without improvement, decreasing learning rate")
        elif scheduler.num_bad_epochs == 20:
            logger.info(f"  20 epochs without improvement, early stopping")
            break

if __name__ == "__main__":
    config = get_config()
    train(config)
