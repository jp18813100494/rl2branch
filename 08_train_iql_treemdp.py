import pathlib
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os.path as osp
import sys
import datetime
import numpy as np
import torch
import wandb
import argparse
import json
import utilities
import glob
import pprint
from algos.iql_agent import IQL
from algos.awac_agent import AWAC
from utilities import BuildFullTransition,get_lr
from agent import AgentPool
from scipy.stats.mstats import gmean

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--config', type=str, default='config/setcover/config_iql_mdp.json', help='path to yaml config file')
    parser.add_argument("--algo_name", type=str, default="IQL", help="Run name, default: IQL")
    parser.add_argument("--hidden_size", type=int, default=64, help="")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="actor learning_rate")
    parser.add_argument("--critic_lr", type=float, default=3e-4, help="critic learning_rate")
    parser.add_argument("--actor_on_lr", type=float, default=1e-6, help="actor learning_rate")
    parser.add_argument("--critic_on_lr", type=float, default=1e-6, help="critic learning_rate")
    parser.add_argument("--temperature", type=float, default=3, help="")
    parser.add_argument("--expectile", type=float, default=0.7, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--entropy_bonus", type=float, default=1e-5, help="")
    parser.add_argument("--gamma", type=float, default=0.99, help="")
    parser.add_argument("--lammbda", type=float, default=3.0, help="")
    parser.add_argument("--num_action_samples", type=int, default=1, help="")
    parser.add_argument("--use_adv", type=bool, default=True, help="")
    parser.add_argument("--hard_update_every", type=int, default=10, help="")
    parser.add_argument("--clip_grad_param", type=int, default=100, help="")

    parser.add_argument("--max_epochs", type=int, default=1000, help="Number of max_epochs, default: 1000")
    parser.add_argument("--save_every", type=int, default=10, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--eval_every", type=int, default=10, help="")
    parser.add_argument("--eval_run", type=int, default=5, help="")
    parser.add_argument("--num_workers", type=int, default=16, help="")
    parser.add_argument("--num_valid_seeds", type=int, default=5, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default: 256")
    parser.add_argument("--valid_batch_size", type=int, default=128, help="Valid Batch size, default: 256")
    parser.add_argument("--num_valid_instances", type=int, default=20, help="Number of valid instances for branch_env")
    parser.add_argument("--num_train_samples", type=int, default=500, help="Number of valid instances for branch_env")
    parser.add_argument("--num_episodes_per_epoch", type=int, default=10, help="Number of train samples in every epoch")
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
        out_dir = 'data/samples_{}/setcover/400r_750c_0.05d'.format(args.mode)
        time_limit = None
    elif config['problem'] == "cauctions":
        maximization = True
        valid_path = "data/instances/cauctions/valid_100_500"
        train_path = "data/instances/cauctions/train_100_500"
        out_dir = 'data/samples_{}/cauctions/100_500'.format(args.mode)
        time_limit = None
    elif config['problem'] == "indset":
        maximization = True
        valid_path = "data/instances/indset/valid_500_4"
        train_path = "data/instances/indset/train_500_4"
        out_dir = 'data/samples_{}/indset/500_4'.format(args.mode)
        time_limit = None
    elif config['problem'] == "ufacilities":
        maximization = False
        valid_path = "data/instances/ufacilities/valid_35_35_5"
        train_path = "data/instances/ufacilities/train_35_35_5"
        out_dir = 'data/samples_{}/ufacilities/35_35_5'.format(args.mode)
        time_limit = None
    elif config['problem'] == "mknapsack":
        maximization = True
        valid_path = "data/instances/mknapsack/valid_100_6"
        train_path = "data/instances/mknapsack/train_100_6"
        out_dir = 'data/samples_{}/mknapsack/100_6'.format(args.mode)
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
    os.system('cp -r config algos envs *.json  *.py {} {}'.format(args.config, osp.join(config["model_dir"], 'code')))

def train(config, args):
    torch.manual_seed(config['seed'])
    sys.path.insert(0, os.path.abspath(f'./results'))
    rng = np.random.RandomState(config['seed'])
    ### LOG ###
    logger = utilities.configure_logging()
    mkdirs(config,args,logger)
    if config['wandb']:
        wandb.init(project="rl2branch", name=config['cur_name'], config=config)

    # recover training data & validation instances
    train_files = [str(file) for file in (pathlib.Path(f'data/samples_{args.mode}')/config['problem_folder']/'train').glob('sample_*.pkl')]
    valid_path = config['valid_path']
    train_path = config['train_path']
    valid_instances = [f'{valid_path}/instance_{j+1}.lp' for j in range(config['num_valid_instances'])]
    train_instances = [f'{train_path}/instance_{j+1}.lp' for j in range(len(glob.glob(f'{train_path}/instance_*.lp')))]
    valid_batch = [{'path': instance, 'seed': seed} for instance in valid_instances for seed in range(config['num_valid_seeds'])]
    # collect the pre-computed optimal solutions for the training instances
    with open(f"{config['train_path']}/instance_solutions.json", "r") as f:
        train_sols = json.load(f)
    with open(f"{config['valid_path']}/instance_solutions.json", "r") as f:
        valid_sols = json.load(f)
    eps = config["eps"] = -0.1 if config['maximization'] else 0.1

    def train_batch_generator():
        while True:
            yield [{'path': instance, 'sol': train_sols[instance] + eps, 'seed': rng.randint(0, 2**32)}
                    for instance in rng.choice(train_instances, size=config['num_episodes_per_epoch'], replace=True)]
    train_batches = train_batch_generator()
    logger.info(f"Training on {len(train_files)} training instances and {len(valid_instances)} validation instances")

    batches = 0
    if config['algo_name'] == 'IQL':
        agent = IQL(config=config)
        logger.info(f'Epoch:0, Actor_lr:{get_lr(agent.actor_optimizer)},Critic_lr:{get_lr(agent.value_optimizer)}')
    elif config['algo_name'] == 'AWAC':
        agent = AWAC(config=config)
        logger.info(f'Epoch:0, Actor_lr:{get_lr(agent.actor_optimizer)},Critic_lr:{get_lr(agent.critic1_optimizer)}')
    else:
        logger.info('Provide the exact algorithm name')
    agent_pool = AgentPool(agent, config['num_workers'], config['time_limit'], config["mode"])
    agent_pool.start()
    
    is_validation_epoch = lambda epoch: (epoch % config['eval_every'] == 0) or (epoch == config['max_epochs'])
    is_offline_training_epoch = lambda epoch: (epoch < config['max_epochs']) and (config['train_stat'] == 'offline')
    is_online_training_epoch = lambda epoch: (epoch < config['max_epochs']) and (config['train_stat'] == 'online')

    # Already start jobs
    if is_validation_epoch(0):
         _, v_stats_next, v_queue_next, v_access_next = agent_pool.start_job(valid_batch, sample_rate=0.0, greedy=True, block_policy=True)
    if is_online_training_epoch(0):
        train_batch = next(train_batches)
        t_samples_next, t_stats_next, t_queue_next, t_access_next = agent_pool.start_job(train_batch, sample_rate=config['sample_rate'], greedy=False, block_policy=True)
    
    config['best_tree_size'] = np.inf
    for epoch in range(0, config['max_epochs']+1):
        logger.info(f'** Epoch {epoch}')
        wandb_data = {}
        if epoch==5:
            print('Stop!!!')
        # Allow preempted jobs to access policy
        if is_validation_epoch(epoch):
            v_stats, v_queue, v_access = v_stats_next, v_queue_next, v_access_next
            v_access.set()
            logger.info(f"  {len(valid_batch)} validation jobs running (preempted)")
            # do not do anything with the stats yet, we have to wait for the jobs to finish !
        else:
            logger.info(f" validation skipped")
        if is_online_training_epoch(epoch):
            t_samples, t_stats, t_queue, t_access = t_samples_next, t_stats_next, t_queue_next, t_access_next
            t_access.set()
            logger.info(f"Status: {config['train_stat']}-{len(train_batch)} training jobs running (preempted)")
        if is_offline_training_epoch(epoch):
            epoch_train_files = rng.choice(train_files, config['hard_update_every']*config['batch_size'], replace=True)
            t_samples,t_stats = BuildFullTransition(epoch_train_files)
            logger.info(f"Status: {config['train_stat']}-{len(t_samples)} training samples was selected")

        # Start next epoch's jobs
        if epoch + 1 <= config["max_epochs"]:
            if is_validation_epoch(epoch+1):
                _, v_stats_next, v_queue_next, v_access_next = agent_pool.start_job(valid_batch, sample_rate=0.0, greedy=True, block_policy=True)
            if is_online_training_epoch(epoch+1):
                train_batch = next(train_batches)
                t_samples_next, t_stats_next, t_queue_next, t_access_next = agent_pool.start_job(train_batch, sample_rate=config['sample_rate'], greedy=False, block_policy=True)    

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
                agent.save(wandb, config['train_stat'])
        # Training
        if is_online_training_epoch(epoch) or is_offline_training_epoch(epoch):
            if config['train_stat'] == 'online':
                t_queue.join()  # wait for all training episodes to be processed
            logger.info('training jobs finished')
            logger.info(f" {len(t_samples)} training samples collected")
            t_losses = agent.update(t_samples)
            logger.info(' model parameters were updated')

            wandb_data.update({
                'train_nsamples': len(t_samples),
                'train_actor_loss': t_losses.get('actor_loss', None),
                'train_loss': t_losses.get('loss', None),
                'train_entropy': t_losses.get('entropy', None),
                'train_critic1_loss': t_losses.get('critic1_loss', None),
                'train_critic2_loss': t_losses.get('critic2_loss', None),
                'train_value_loss': t_losses.get('value_loss', None),
                'actor_lr':get_lr(agent.actor_optimizer),
            })
            logger.info(f"Episode: {epoch} | Batches: {batches} | Polciy Loss: {t_losses.get('actor_loss', None)}  | Critic Loss: {t_losses.get('critic1_loss', None)} | Value Loss: {t_losses.get('value_loss', 0)} ")

        # Send the stats to wandb
        if config['wandb']:
            wandb.log(wandb_data, step = epoch)
        if epoch % config['save_every'] == 0 and config['wandb']:
            agent.save(wandb=wandb, stat=config['train_stat'], ep=epoch)
        
        config["cur_epoch"] = epoch
        if is_validation_epoch(epoch) and config['train_stat'] == 'offline':
            agent.scheduler_step(wandb_data['valid_nnodes_g'])
            if config['wandb'] and agent.actor_scheduler.num_bad_epochs == 0:
                logger.info(f"best model so far")
            elif agent.actor_scheduler.num_bad_epochs == 5:
                logger.info(f"5 epochs without improvement, decreasing learning rate")
            elif agent.actor_scheduler.num_bad_epochs == 10:
                logger.info(f"Offline: 10 epochs without improvement, switch to online")
                logger.info(f'Offline for {epoch} epochs')
                config['train_stat'] = 'online'
                logger.info(f'Start online training')
                agent.reset_optimizer()
                train_batch = next(train_batches)
                t_samples_next, t_stats_next, t_queue_next, t_access_next = agent_pool.start_job(train_batch, sample_rate=config['sample_rate'], greedy=False, block_policy=True)
                # else:
                #     logger.info(f"Online:10 epochs without improvement, early stopping")
                #     break

    if config["wandb"]:
        wandb.join()
        wandb.finish()
    v_access_next.set()
    t_access_next.set()
    agent_pool.close()
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    config,args = get_config()
    train(config,args)
