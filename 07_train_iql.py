import pathlib
import os
import shutil
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
from utilities import Scheduler,BuildFullTransition,evaluate,wandb_eval_log
from envs.branch_env import branch_env
from collect_samples import collect_online

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--config', type=str, default='config/setcover/config_iql_mdp.json', help='path to yaml config file')
    parser.add_argument("--algo_name", type=str, default="IQL", help="Run name, default: SAC")
    parser.add_argument("--hidden_size", type=int, default=64, help="")
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="actor learning_rate")
    parser.add_argument("--critic_lr", type=float, default=3e-4, help="critic learning_rate")
    parser.add_argument("--temperature", type=float, default=3, help="")
    parser.add_argument("--expectile", type=float, default=0.7, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--gamma", type=float, default=0.99, help="")
    parser.add_argument("--hard_update_every", type=int, default=10, help="")
    parser.add_argument("--clip_grad_param", type=int, default=100, help="")

    parser.add_argument("--max_epochs", type=int, default=100, help="Number of max_epochs, default: 1000")
    parser.add_argument("--save_every", type=int, default=10, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--eval_every", type=int, default=3, help="")
    parser.add_argument("--eval_run", type=int, default=5, help="")
    # parser.add_argument("--num_workers", type=int, default=8, help="")
    parser.add_argument("--num_valid_seeds", type=int, default=12, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default: 256")
    parser.add_argument("--valid_batch_size", type=int, default=128, help="Valid Batch size, default: 256")
    parser.add_argument("--num_valid_instances", type=int, default=1000, help="Number of valid instances for branch_env")
    parser.add_argument("--epoch_train_size", type=int, default=1000, help="Number of train samples in every epoch")
    parser.add_argument("--njobs", type=int, default=12, help='Number of parallel jobs.')
    parser.add_argument("--num_repeat", type=int, default=5, help='Number of repeat for sample data')
    parser.add_argument("--node_record_prob", type=float, default=1.0, help='Probability for recording tree nodes')
    parser.add_argument("--init_stat", type=str, default='offline', choices=['offline', 'online'], help="Init stat for agent")
    
    parser.add_argument('--problem',help='MILP instance type to process.',choices=['setcover', 'cauctions', 'ufacilities', 'indset', 'mknapsack'],default='setcover',)
    parser.add_argument("--mode", type=str, default='mdp', choices=['mdp', 'tmdp+ObjLim', 'tmdp+DFS'], help="Mode for branch env")
    parser.add_argument("--time_limit", type=int, default=4000, help="Timelimit of the solver")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="")

    parser.add_argument('--seed',help='Random generator seed.',type=int,default=0)
    parser.add_argument('--gpu',help='CUDA GPU id (-1 for CPU).',type=int,default=0)
    parser.add_argument('--wandb',help="Use wandb?",default=False,action="store_true")
    args = parser.parse_args()

    # override args with the user args file if provided
    args.config = osp.realpath(args.config)
    with open(args.config, 'r') as f:
        config = json.load(f)

    # override args with command-line arguments if provided
    args_config = {key: getattr(args, key) for key in config.keys() & vars(args).keys()}
    config.update(args_config)

    config['hard_update_every'] = int(np.floor(config['num_train_samples']/config['batch_size']))
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
        out_dir = 'data/samples_orl/setcover/400r_750c_0.05d'
        time_limit = None
    elif config['problem'] == "cauctions":
        maximization = True
        valid_path = "data/instances/cauctions/valid_100_500"
        train_path = "data/instances/cauctions/train_100_500"
        out_dir = 'data/samples_orl/cauctions/100_500'
        time_limit = None
    elif config['problem'] == "indset":
        maximization = True
        valid_path = "data/instances/indset/valid_500_4"
        train_path = "data/instances/indset/train_500_4"
        out_dir = 'data/samples_orl/indset/500_4'
        time_limit = None
    elif config['problem'] == "ufacilities":
        maximization = False
        valid_path = "data/instances/ufacilities/valid_35_35_5"
        train_path = "data/instances/ufacilities/train_35_35_5"
        out_dir = 'data/samples_orl/ufacilities/35_35_5'
        time_limit = 600
    elif config['problem'] == "mknapsack":
        maximization = True
        valid_path = "data/instances/mknapsack/valid_100_6"
        train_path = "data/instances/mknapsack/train_100_6"
        out_dir = 'data/samples_orl/mknapsack/100_6'
        time_limit = 60
    else:
        raise NotImplementedError
    config['train_path'] = train_path
    config['valid_path'] = valid_path
    config['out_dir'] = out_dir
    config['time_limit'] = time_limit if time_limit is not None else config["time_limit"]
    config['maximization'] = maximization
        # model path
    cur_name = '{}-{}-{}-{}'.format(config['algo_name'],  config['mode'], config['problem'],  datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    token = '{}/{}'.format(config['problem'], cur_name)
    model_dir = osp.realpath(osp.join('results', token))
    config['model_dir'] = model_dir
    config['cur_name'] = cur_name

    problem_folders = {
        'setcover': 'setcover/400r_750c_0.05d',
        'cauctions': 'cauctions/100_500',
        'ufacilities': 'ufacilities/35_35_5',
        'indset': 'indset/500_4',
        'mknapsack': 'mknapsack/100_6',
    }
    config['problem_folder'] = problem_folders[config['problem']]
    return config



def train(config):
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    sys.path.insert(0, os.path.abspath(f'./results'))

    ### LOG ###
    logger = utilities.configure_logging()
    if config['wandb']:
        wandb.init(project="rl2branch", name=config['cur_name'], config=config)

    is_validation_epoch = lambda epoch: (epoch % config['eval_every'] == 0) or (epoch == config['max_epochs'])
    is_training_epoch = lambda epoch: (epoch < config['max_epochs'])
    # recover training data & validation instances
    train_files = [str(file) for file in (pathlib.Path(f'data/samples_orl')/config['problem_folder']/'train').glob('sample_*.pkl')]
    valid_path = config['valid_path']
    train_path = config['train_path']
    valid_instances = [f'{valid_path}/instance_{j+1}.lp' for j in range(config['num_valid_instances'])]
    train_instances = [f'{train_path}/instance_{j+1}.lp' for j in range(len(glob.glob(f'{train_path}/instance_*.lp')))]
    logger.info(f"Training on {len(train_files)} training instances and {len(valid_instances)} validation instances")
    # collect the pre-computed optimal solutions for the training instances
    with open(f"{config['train_path']}/instance_solutions.json", "r") as f:
         train_sols = json.load(f)
    with open(f"{config['valid_path']}/instance_solutions.json", "r") as f:
        valid_sols = json.load(f)
    env = branch_env(valid_instances,valid_sols,config)

    batches = 0
    stat = config['init_stat']
    average10 = deque(maxlen=10)
    
    agent = IQL(state_size=env.observation_space.shape[0],
                action_size=env.action_space.shape[0],
                actor_lr=config['actor_lr'],
                critic_lr=config['critic_lr'],
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
  
    v_reward,_ = evaluate(env, agent,config['eval_run'], logger)
    logger.info(f"Test Reward: {v_reward}, Episode: 0, Batches: {batches}")
    config['best_tree_size'] = np.inf
    for epoch in range(0, config['max_epochs']+1):
        logger.info(f'** Epoch {epoch}')
        wandb_data = {}
        if stat == 'offline':
            epoch_train_files = rng.choice(train_files, config['hard_update_every']*config['batch_size'], replace=True)
            train_data = BuildFullTransition(epoch_train_files)
        else:
            tmp_samples_dir = f'{config["out_dir"]}/train/tmp'
            os.makedirs(tmp_samples_dir, exist_ok=True)
            epoch_train_files = collect_online(train_instances, tmp_samples_dir, rng, config["epoch_train_size"],
                    config["njobs"], query_expert_prob=config["node_record_prob"],
                    time_limit=config["time_limit"], agent=agent)
            epoch_train_files = config['num_repeat']*epoch_train_files
            train_data = BuildFullTransition(epoch_train_files)
            shutil.rmtree(tmp_samples_dir, ignore_errors=True)
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

        if is_validation_epoch(epoch):
            v_reward, v_stats = evaluate(env, agent,config['eval_run'], logger)
            wandb_data = wandb_eval_log(epoch, agent, wandb, wandb_data, v_stats, v_reward, config, logger)
            average10.append(v_reward)
        logger.info(f"Episode: {epoch} | Batches: {batches} | Polciy Loss: {np.mean(policy_losses)}  | Value Loss: {np.mean(value_losses)} | Critic Loss: {np.mean(critic1_losses)} ")
        if is_training_epoch(epoch):
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
            save(config,  model=agent, wandb=wandb, stat='offline', ep=epoch)
        
        config["cur_epoch"] = epoch
        scheduler.step(np.mean(policy_losses))
        if config['wandb'] and scheduler.num_bad_epochs == 0:
            torch.save(agent.actor_local.state_dict(), pathlib.Path(config['model_dir'])/'iql_best_actor.pkl')
            logger.info(f"best model so far")
        elif scheduler.num_bad_epochs == 10:
            logger.info(f"10 epochs without improvement, decreasing learning rate")
        elif scheduler.num_bad_epochs == 20:
            logger.info(f"20 epochs without improvement, early stopping")
            if stat == 'offline':
                stat = 'online'
            else:
                break
        

    if config["wandb"]:
        wandb.join()
        wandb.finish()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    config = get_config()
    train(config)
