# from calendar import c
# from email import policy
# from re import M
import torch
import torch_geometric
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from algos.network import Critic, Actor, Value


class IQL(nn.Module):
    def __init__(self,config): 
        super(IQL, self).__init__()
        self.config = config
        self.device = config["device"]
        self.actor_lr = config["actor_lr"]
        self.critic_lr = config["critic_lr"]
        self.tau = config["tau"]
        self.gamma = config["gamma"]
        self.hard_update_every = config["hard_update_every"]
        self.clip_grad_param = config["clip_grad_param"]
        self.temperature = config["temperature"]
        self.expectile = config["expectile"]
        self.seed=config["seed"]
        self.actor_on_lr = config["actor_on_lr"]
        self.critic_on_lr = config["critic_on_lr"]
        self.hidden_size = config['hidden_size']
        self.demonstrator_margin = config['demonstrator_margin']
        self.sl_loss_factor = config["sl_loss_factor"]
        
        # Actor Network 
        self.actor_local = Actor(self.hidden_size, self.seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_lr)     
        
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(self.hidden_size, 2).to(self.device)
        self.critic2 = Critic(self.hidden_size, 1).to(self.device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(self.hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(self.hidden_size).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic_optimizer = optim.Adam([
                                                                {'params': self.critic1.parameters(), 'lr': self.critic_lr}, 
                                                                {'params': self.critic2.parameters(), 'lr': self.critic_lr}
                                                                ])
        
        self.value_net = Value(self.hidden_size).to(self.device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.critic_lr)
        self.step = 0

        self.actor_scheduler = Scheduler(self.actor_optimizer, mode='min', patience=15, factor=0.2, verbose=True)
        self.critic_scheduler = Scheduler(self.critic_optimizer, mode='min', patience=15, factor=0.2, verbose=True)
        self.value_scheduler = Scheduler(self.value_optimizer, mode='min', patience=15, factor=0.2, verbose=True)
        
        # full_layers = ['cons_embedding','edge_embedding','var_embedding','conv_v_to_c','conv_c_to_v','output_module']
        full_layers = ['cons_embedding','edge_embedding','var_embedding']
        self.freeze_layers = []
        for i in range(self.config['pretrain_module_nums']):
            self.freeze_layers.append(full_layers[i])

    def scheduler_step(self,metric):
        self.actor_scheduler.step(metric)
        self.critic_scheduler.step(metric)
        self.value_scheduler.step(metric)
        

    def reset_optimizer(self):
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_on_lr) 
        self.critic_optimizer = optim.Adam([
                                                                {'params': self.critic1.parameters(), 'lr': self.critic_on_lr}, 
                                                                {'params': self.critic2.parameters(), 'lr': self.critic_on_lr}
                                                                ])
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.critic_on_lr)

    def sample_action_idx(self, states, greedy):
        if isinstance(greedy, bool):
            greedy = torch.tensor(np.repeat(greedy, len(states), dtype=torch.long))
        elif not isinstance(greedy, torch.Tensor):
            greedy = torch.tensor(greedy, dtype=torch.long)

        states_loader = torch_geometric.data.DataLoader(states, batch_size=self.config['batch_size'])
        greedy_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(greedy), batch_size=self.config['batch_size'])
        action_idxs = []
        for batch, (greedy,) in zip(states_loader, greedy_loader):
            with torch.no_grad():
                batch = batch.to(self.device)
                logits = self.actor_local(batch, greedy)

                logits_end = batch.action_set_size.cumsum(-1)
                logits_start = logits_end - batch.action_set_size
                for start, end, greedy in zip(logits_start, logits_end, greedy):
                    if greedy:
                        action_idx = logits[start:end].argmax()
                    else:
                        action_idx = torch.distributions.categorical.Categorical(logits=logits[start:end]).sample()
                    action_idxs.append(action_idx.item())

        return action_idxs

    def get_action(self, state, eval=False,num_samples=1):
        """Returns actions for given state as per current policy."""
        state = state.to(self.device)
        with torch.no_grad():
            action = self.actor_local.get_action(state, eval,num_samples)
        return action.detach().cpu()

    def calc_policy_loss(self, states, actions):
        with torch.no_grad():
            v = self.value_net(states)
            q1 = self.critic1_target(states).gather(1, actions.long())
            q2 = self.critic2_target(states).gather(1, actions.long())
            min_Q = torch.min(q1,q2)

        exp_a = torch.exp((min_Q - v) * self.temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(self.device)).squeeze(-1)

        _, dist, logits = self.actor_local.evaluate(states)
        log_probs = dist.log_prob(actions.squeeze(-1))
        actor_loss = -(exp_a * log_probs).mean()
        entropy = dist.entropy().sum()
        if self.sl_loss_factor>0:
            policy_sl_loss = self.calc_policy_sl_loss(logits, actions)
        else:
            policy_sl_loss = 0

        return actor_loss,entropy,policy_sl_loss
    
    def calc_value_loss(self, states, actions):
        with torch.no_grad():
            q1 = self.critic1_target(states).gather(1, actions.long())
            q2 = self.critic2_target(states).gather(1, actions.long())
            min_Q = torch.min(q1,q2)
        
        value = self.value_net(states)
        value_loss = loss(min_Q - value, self.expectile).mean()
        return value_loss
    
    def calc_q_loss(self, states, actions, rewards, dones, next_states,tree=False):
        dones = torch.unsqueeze(dones,-1)
        rewards = torch.unsqueeze(rewards,-1)
        with torch.no_grad():
            if tree:
                next_states_l,next_states_r = next_states
                next_l_v = self.value_net(next_states_l)
                next_r_v = self.value_net(next_states_r)
                q_target = rewards + (self.gamma * (1 - dones) * (next_l_v+next_r_v)) 
            else:
                next_v = self.value_net(next_states)
                q_target = rewards + (self.gamma * (1 - dones) * next_v) 
        q1_values = self.critic1(states)
        q2_values = self.critic2(states)
        q1 = q1_values.gather(1, actions.long())
        q2 = q2_values.gather(1, actions.long())
        critic1_loss = ((q1 - q_target)**2).mean() 
        critic2_loss = ((q2 - q_target)**2).mean()
        if self.sl_loss_factor>0:
            q1_sl_loss = self.calc_q_sl_loss(q1_values,actions)
            q2_sl_loss = self.calc_q_sl_loss(q2_values,actions)
            critic1_loss +=  self.sl_loss_factor*q1_sl_loss.mean()
            critic2_loss +=  self.sl_loss_factor*q2_sl_loss.mean()
        return critic1_loss, critic2_loss

    def calc_q_sl_loss(self, q_values, actions):
        agent_action_idxs = torch.stack([q.argmax() for q in q_values]).unsqueeze(-1)
        margin_function = torch.where(agent_action_idxs == actions, 0, 1) * self.demonstrator_margin
        sl_loss = (q_values.gather(1, agent_action_idxs.long())+ margin_function) - q_values.gather(1, actions.long())
        return sl_loss
    
    def calc_policy_sl_loss(self, logits,actions):
        cross_entropy_loss = F.cross_entropy(logits, actions.squeeze(), reduction='mean')
        return cross_entropy_loss

    def update(self, transitions):
        n_samples = len(transitions)
        if n_samples < 1:
           stats = {'loss': 0.0, 'actor_loss': 0.0, 'critic1_loss': 0.0, 'critic2_loss': 0.0, 'value_loss': 0.0 }
           return stats

        transitions = torch_geometric.data.DataLoader(transitions, batch_size=self.config["batch_size"], shuffle=True)
        stats = {}

        self.value_optimizer.zero_grad()
        for batch in transitions:
            self.step += 1
            batch = batch.to(self.device)
            states = (batch.constraint_features, batch.edge_index, batch.edge_attr,batch.variable_features,batch.action_set,batch.action_set_size)
            actions = batch.action_idx.unsqueeze(1)
            value_loss = self.calc_value_loss(states, actions)
            value_loss /= n_samples
            value_loss.backward()
            # Update stats
            stats['value_loss'] = stats.get('value_loss', 0.0) + value_loss.item()
        self.value_optimizer.step()

        self.actor_optimizer.zero_grad()
        for batch in transitions:
            batch = batch.to(self.device)
            loss = torch.tensor([0.0], device=self.device)
            states = (batch.constraint_features, batch.edge_index, batch.edge_attr,batch.variable_features,batch.action_set,batch.action_set_size)
            actions = batch.action_idx.unsqueeze(1)
            actor_loss,entropy,policy_sl_loss = self.calc_policy_loss(states, actions)
            actor_loss /= n_samples
            loss += actor_loss
            entropy /= n_samples
            loss += - self.config['entropy_bonus']*entropy
            if self.sl_loss_factor>0:
                policy_sl_loss /= n_samples
                loss += self.sl_loss_factor*policy_sl_loss
            loss.backward()
            # Update stats
            stats['actor_loss'] = stats.get('actor_loss', 0.0) + actor_loss.item()
            stats['loss'] = stats.get('loss', 0.0) + loss.item()
            stats['entropy'] = stats.get('entropy', 0.0) + entropy.item()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        for batch in transitions:
            batch = batch.to(self.device)
            states = (batch.constraint_features, batch.edge_index, batch.edge_attr,batch.variable_features,batch.action_set,batch.action_set_size)
            actions = batch.action_idx.unsqueeze(1)
            rewards = batch.reward
            if batch.tree[0]:
                next_states_l = (batch.constraint_features_l, batch.edge_index_l, batch.edge_attr_l, 
                            batch.variable_features_l,batch.action_set_l,batch.action_set_l_size)
                next_states_r = (batch.constraint_features_r, batch.edge_index_r, batch.edge_attr_r, 
                            batch.variable_features_r,batch.action_set_r,batch.action_set_r_size)
                next_states = (next_states_l,next_states_r)
            else:
                next_states = (batch.constraint_features_n, batch.edge_index_n, batch.edge_attr_n, 
                                        batch.variable_features_n,batch.action_set_n,batch.action_set_n_size)
            dones = batch.done
            critic1_loss, critic2_loss = self.calc_q_loss(states, actions, rewards, dones, next_states, batch.tree[0])
            critic1_loss /= n_samples
            critic1_loss.backward()
            clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)

            critic2_loss /= n_samples
            critic2_loss.backward()
            clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
            # Update stats
            stats['critic1_loss'] = stats.get('critic1_loss', 0.0) + critic1_loss.item()
            stats['critic2_loss'] = stats.get('critic2_loss', 0.0) + critic2_loss.item()
        self.critic_optimizer.step()

        if self.step % self.hard_update_every == 0:
            # ----------------------- update target networks ----------------------- #
            self.hard_update(self.critic1, self.critic1_target)
            self.hard_update(self.critic2, self.critic2_target)
        return stats


    # def learn(self, batch):
    #     self.step += 1
    #     states = (batch.constraint_features, batch.edge_index, batch.edge_attr, 
    #                 batch.variable_features,batch.action_set,batch.action_set_size)
    #     actions = batch.action_idx.unsqueeze(1)
    #     rewards = batch.reward
    #     next_states = (batch.constraint_features_n, batch.edge_index_n, batch.edge_attr_n, 
    #                     batch.variable_features_n,batch.action_set_n,batch.action_set_n_size)
    #     dones = batch.done

    #     # assert batch.edge_index.max()<batch.variable_features.shape[0]
    #     # assert batch.edge_index_n.max()<batch.variable_features_n.shape[0]
        
    #     # states, actions, rewards, next_states, dones = batch        
    #     self.value_optimizer.zero_grad()
    #     value_loss = self.calc_value_loss(states, actions)
    #     value_loss.backward()
    #     self.value_optimizer.step()

    #     actor_loss,_ = self.calc_policy_loss(states, actions)
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     self.actor_optimizer.step()
        
    #     critic1_loss, critic2_loss = self.calc_q_loss(states, actions, rewards, dones, next_states)

    #     # critic 1
    #     self.critic1_optimizer.zero_grad()
    #     critic1_loss.backward()
    #     clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
    #     self.critic1_optimizer.step()
    #     # critic 2
    #     self.critic2_optimizer.zero_grad()
    #     critic2_loss.backward()
    #     clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
    #     self.critic2_optimizer.step()

    #     if self.step % self.hard_update_every == 0:
    #         # ----------------------- update target networks ----------------------- #
    #         self.hard_update(self.critic1, self.critic1_target)
    #         self.hard_update(self.critic2, self.critic2_target)
        
    #     return actor_loss.item(), critic1_loss.item(), critic2_loss.item(), value_loss.item()
    
    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
        

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    
    def save(self, wandb, stat="offline",ep=None):
        import os
        models_dir = os.path.join(self.config['model_dir'],"models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        if not ep == None:
            torch.save(self.actor_local.state_dict(), models_dir +'/actor_'+ stat + str(ep) + ".pth")
            # wandb.save(models_dir +'/actor_'+ str(ep) + ".pth")
            torch.save(self.critic1.state_dict(), models_dir +'/critic1_'+ stat + str(ep) + ".pth")
            torch.save(self.critic2.state_dict(), models_dir +'/critic2_'+ stat + str(ep) + ".pth")
            # wandb.save(models_dir +'/critic1_'+ str(ep) + ".pth")
            torch.save(self.value_net.state_dict(), models_dir +'/value_'+ stat + str(ep) + ".pth")
            # wandb.save(models_dir +'/critic1_'+ str(ep) + ".pth")
        else:
            torch.save(self.actor_local.state_dict(), models_dir +'/actor_'+ stat + "best.pth")
            # wandb.save(models_dir +'/actor_'+ "best.pth")
            torch.save(self.critic1.state_dict(), models_dir +'/critic1_'+ stat + "best.pth")
            torch.save(self.critic2.state_dict(), models_dir +'/critic2_'+ stat + "best.pth")
            # wandb.save(models_dir +'/critic1_'+ "best.pth")
            torch.save(self.value_net.state_dict(), models_dir +'/value_' +stat+ "best.pth")
            # wandb.save(models_dir +'/value_'+ "best.pth")
    
    def load_model(self,path):
        self.actor_local.load_state_dict(torch.load(path+'/actor_offlinebest.pth'))
        self.critic1.load_state_dict(torch.load(path+'/critic1_offlinebest.pth'))
        # self.critic2.load_state_dict(torch.load(path+'/critic1_offline90.pth'))
        self.critic2.load_state_dict(torch.load(path+'/critic2_offlinebest.pth'))
        self.value_net.load_state_dict(torch.load(path+'/value_offlinebest.pth'))

    def freeze_layer(self, model):
        for name,param in model.named_parameters():
            # print(name,param.shape)
            if name.split(".")[0] in self.freeze_layers:
                param.require_gard = False

    def freeze_part_model(self):
        self.freeze_layer(self.actor_local)
        self.freeze_layer(self.critic1)
        self.freeze_layer(self.critic2)
        self.freeze_layer(self.value_net)

    def switch_to_online(self):
        self.freeze_part_model()
        self.config['train_stat'] = 'online'
        self.reset_optimizer()
        self.config['entropy_bonus'] = self.config["entropy_bonus_on"]
        self.sl_loss_factor = 0


def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

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