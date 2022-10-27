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
import pprint
from collections import deque
from iql.iql_agent import IQL,save,get_lr
from utilities import Scheduler,BuildFullTransition,evaluate,wandb_eval_log
from envs.branch_env import branch_env
from collect_samples import collect_online
from agent import AgentPool
from scipy.stats.mstats import gmean

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
    parser.add_argument("--num_workers", type=int, default=10, help="")
    parser.add_argument("--num_valid_seeds", type=int, default=5, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default: 256")
    parser.add_argument("--valid_batch_size", type=int, default=128, help="Valid Batch size, default: 256")
    parser.add_argument("--num_valid_instances", type=int, default=20, help="Number of valid instances for branch_env")
    parser.add_argument("--num_train_samples", type=int, default=5000, help="Number of valid instances for branch_env")
    parser.add_argument("--epoch_train_size", type=int, default=2000, help="Number of train samples in every epoch")
    parser.add_argument("--njobs", type=int, default=4, help='Number of parallel jobs.')
    parser.add_argument("--num_repeat", type=int, default=5, help='Number of repeat for sample data')
    parser.add_argument("--node_record_prob", type=float, default=1.0, help='Probability for recording tree nodes')
    parser.add_argument("--query_expert_prob", type=float, default=1.0, help='Probability for query the expert')
    parser.add_argument("--train_stat", type=str, default='offline', choices=['offline', 'online'], help="Init stat for agent")
    
    parser.add_argument('--problem',default='setcover',choices=['setcover', 'cauctions', 'ufacilities', 'indset', 'mknapsack'],help='MILP instance type to process.')
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
        time_limit = None
    elif config['problem'] == "mknapsack":
        maximization = True
        valid_path = "data/instances/mknapsack/valid_100_6"
        train_path = "data/instances/mknapsack/train_100_6"
        out_dir = 'data/samples_orl/mknapsack/100_6'
        time_limit = None
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
    return config, args

def mkdirs(config, args, logger):
    logger.info(f'cwd: {os.getcwd()}, model_dir: {config["model_dir"]}')
    logger.info(f'Pytorch (version {torch.__version__}), path: {torch}')
    logger.info(f'Command line args: {pprint.pformat(vars(args))}')
    logger.info(f'Parsed config from {args.config}')
    os.makedirs(osp.join(config["model_dir"], 'code'))
    os.makedirs(osp.join(config["model_dir"], 'models'))
    os.system('cp -r config iql envs *.json *.sh *.py {} {}'.format(args.config, osp.join(config["model_dir"], 'code')))

def train(config, args):
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    sys.path.insert(0, os.path.abspath(f'./results'))

    ### LOG ###
    logger = utilities.configure_logging()
    mkdirs(config,args,logger)
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
    valid_batch = [{'path': instance, 'seed': seed} for instance in valid_instances for seed in range(config['num_valid_seeds'])]
    # collect the pre-computed optimal solutions for the training instances
    with open(f"{config['train_path']}/instance_solutions.json", "r") as f:
        train_sols = json.load(f)
    with open(f"{config['valid_path']}/instance_solutions.json", "r") as f:
        valid_sols = json.load(f)
    config["eps"] = -0.1 if config['maximization'] else 0.1
    env = branch_env(valid_instances,valid_sols,config)


    batches = 0
    stat = config['train_stat']
    average10 = deque(maxlen=10)
    
    agent = IQL(state_size=env.observation_space.shape[0],
                action_size=env.action_space.shape[0],
                config=config,
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
    logger.info(f'Epoch:0, Actor_lr:{get_lr(agent.actor_optimizer)},Critic_lr:{get_lr(agent.value_optimizer)}')
    agent_pool = AgentPool(agent, config['num_workers'], config['time_limit'], config["mode"])
    agent_pool.start()
    
    rng = np.random.RandomState(config['seed'])
    # env.seed(config["seed"])
    # v_reward,_ = evaluate(env, agent,config['eval_run'], logger)
    # Already start jobs
    if is_validation_epoch(0):
        _, v_stats_next, v_queue_next, v_access_next = agent_pool.start_job(valid_batch, sample_rate=0.0, greedy=True, block_policy=True)
    config['best_tree_size'] = np.inf
    for epoch in range(0, config['max_epochs']+1):
        logger.info(f'** Epoch {epoch}')
        wandb_data = {}
        # Allow preempted jobs to access policy
        if is_validation_epoch(epoch):
            v_stats, v_queue, v_access = v_stats_next, v_queue_next, v_access_next
            v_access.set()
            logger.info(f"  {len(valid_batch)} validation jobs running (preempted)")
            # do not do anything with the stats yet, we have to wait for the jobs to finish !
        else:
            logger.info(f"  validation skipped")

        # Start next epoch's jobs
        if epoch + 1 <= config["max_epochs"]:
            if is_validation_epoch(epoch + 1):
                _, v_stats_next, v_queue_next, v_access_next = agent_pool.start_job(valid_batch, sample_rate=0.0, greedy=True, block_policy=True)
        #TODO:这部分修改为适配treeMDP的并行采样形式        
        if stat == 'offline':
            epoch_train_files = rng.choice(train_files, config['hard_update_every']*config['batch_size'], replace=True)
            train_data,_ = BuildFullTransition(epoch_train_files)
        else:
            tmp_samples_dir = f'{config["out_dir"]}/train/tmp'
            os.makedirs(tmp_samples_dir, exist_ok=True)
            epoch_train_files = collect_online(train_instances, train_sols, tmp_samples_dir, rng, agent,config)
            train_data,_ = BuildFullTransition(epoch_train_files)
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

        # if is_validation_epoch(epoch):
        #     v_reward, v_stats = evaluate(env, agent,config['eval_run'], logger)
        #     wandb_data = wandb_eval_log(epoch, agent, wandb, wandb_data, v_stats, v_reward, config, logger)
        #     average10.append(v_reward)
        # Validation
        if is_validation_epoch(epoch):
            v_queue.join()  # wait for all validation episodes to be processed
            logger.info('  validation jobs finished')
            v_nnodess = [s['info']['nnodes'] for s in v_stats]
            v_lpiterss = [s['info']['lpiters'] for s in v_stats]
            v_times = [s['info']['time'] for s in v_stats]

            wandb_data.update({
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

            if config["wandb"] and wandb_data['valid_nnodes_g'] < config["best_tree_size"]:
                config["best_tree_size"] = wandb_data['valid_nnodes_g']
                logger.info('Best parameters so far (1-shifted geometric mean), saving model.')
                save(config, agent, wandb, stat)

        logger.info(f"Episode: {epoch} | Batches: {batches} | Polciy Loss: {np.mean(policy_losses)}  | Value Loss: {np.mean(value_losses)} | Critic Loss: {np.mean(critic1_losses)} ")
        if is_training_epoch(epoch):
            wandb_data.update({
                "actor_lr":get_lr(agent.actor_optimizer),
                # "Valid_reward10": np.mean(average10),
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
        if is_validation_epoch(epoch):
            agent.scheduler_step(wandb_data['valid_nnodes_g'])
            if config['wandb'] and agent.actor_scheduler.num_bad_epochs == 0:
                logger.info(f"best model so far")
            elif agent.actor_scheduler.num_bad_epochs == 5:
                logger.info(f"5 epochs without improvement, decreasing learning rate")
            elif agent.actor_scheduler.num_bad_epochs == 10:
                logger.info(f"10 epochs without improvement, early stopping")
                if stat == 'offline':
                    logger.info(f'Offline for {epoch} epochs')
                    stat = 'online'
                    logger.info(f'Start online training')
                    agent.reset_optimizer()
                else:
                    break

    if config["wandb"]:
        wandb.join()
        wandb.finish()
    v_access_next.set()
    agent_pool.close()
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    config,args = get_config()
    train(config,args)
