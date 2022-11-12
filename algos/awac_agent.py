from re import M
import torch
import torch_geometric
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from algos.network import Critic, Actor


class AWAC(nn.Module):
    def __init__(self,config): 
        super(AWAC, self).__init__()
        self.config = config
        self.device = config["device"]
        self.actor_lr = config["actor_lr"]
        self.critic_lr = config["critic_lr"]
        self.tau = config["tau"]
        self.gamma = config["gamma"]
        self.hard_update_every = config["hard_update_every"]
        self.clip_grad_param = config["clip_grad_param"]
        self.lammbda = config["lammbda"]
        self.use_adv = config["use_adv"]
        self.num_action_samples=config["num_action_samples"]
        self.seed=config["seed"]
        self.actor_on_lr = config["actor_on_lr"]
        self.critic_on_lr = config["critic_on_lr"]
        self.hidden_size = config['hidden_size']

           
        # Actor Network 
        self.actor_local = Actor(self.hidden_size, self.seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_lr)     
        
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(self.hidden_size, 2).to(self.device)
        self.critic1_target = Critic(self.hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.step = 0

        self.actor_scheduler = Scheduler(self.actor_optimizer, mode='min', patience=5, factor=0.2, verbose=True)
        self.critic1_scheduler = Scheduler(self.critic1_optimizer, mode='min', patience=5, factor=0.2, verbose=True)


    def scheduler_step(self,metric):
        self.actor_scheduler.step(metric)
        self.critic1_scheduler.step(metric)       

    def reset_optimizer(self):
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_on_lr) 
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_on_lr)

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
        if num_samples==1:
            state = state.to(self.device)
        with torch.no_grad():
            action = self.actor_local.get_action(state, eval, num_samples)
        if num_samples == 1:    
            return action.detach().cpu()
        else:
            return action

    def calc_policy_loss(self, states, actions):
        logits = self.actor_local(states)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions.squeeze(-1)).view(-1,1)
        entropy = dist.entropy().sum()
        with torch.no_grad():
            if self.use_adv:
                qs = self.critic1_target(states)  # [#. samples x # actions]
                action_probs = F.softmax(logits, dim=-1)
                vs = (qs * action_probs).sum(dim=-1, keepdims=True)
                qas = qs.gather(1, actions.long())
                adv = qas - vs
            else:
                adv = self.critic1_target(states).gather(1, actions.long())

            weight_term = torch.exp(1.0 / self.lammbda * adv)
        actor_loss = (log_prob * weight_term).mean() * -1
        return actor_loss,entropy
    
    def calc_q_loss(self, states, actions, rewards, dones, next_states):
        dones = torch.unsqueeze(dones,-1)
        rewards = torch.unsqueeze(rewards,-1)
        with torch.no_grad():
            if isinstance(next_states,tuple):
                next_states_l,next_states_r = next_states
                #left nodes value func
                qs_l = self.critic1_target(next_states_l)  # [minibatch size x #.actions]
                sampled_as_l = self.get_action(next_states_l, eval=False, num_samples=self.num_action_samples)  # [ minibatch size x #. action samples]
                mean_qsa_l = qs_l.gather(1, sampled_as_l).mean(dim=-1, keepdims=True)  # [minibatch size x 1]
                #right nodes value func
                qs_r = self.critic1_target(next_states_r)  # [minibatch size x #.actions]
                sampled_as_r = self.get_action(next_states_r, eval=False, num_samples=self.num_action_samples)  # [ minibatch size x #. action samples]
                mean_qsa_r = qs_r.gather(1, sampled_as_r).mean(dim=-1, keepdims=True)  # [minibatch size x 1]
                q_target = rewards + self.gamma * (1 - dones) * (mean_qsa_l+mean_qsa_r)
            else:
                qs = self.critic1_target(next_states)  # [minibatch size x #.actions]
                sampled_as = self.get_action(next_states, eval=False, num_samples=self.num_action_samples)  # [ minibatch size x #. action samples]
                mean_qsa = qs.gather(1, sampled_as).mean(dim=-1, keepdims=True)  # [minibatch size x 1]
                q_target = rewards + self.gamma * mean_qsa * (1 - dones)

        q_val = self.critic1(states).gather(1, actions.long())
        critic_loss = ((q_val - q_target)**2).mean() 
        return critic_loss

    def update(self, transitions):
        n_samples = len(transitions)
        if n_samples < 1:
           stats = {'loss': 0.0, 'actor_loss': 0.0, 'critic1_loss': 0.0, 'critic2_loss': 0.0, 'value_loss': 0.0 }
           return stats

        transitions = torch_geometric.data.DataLoader(transitions, batch_size=self.config["batch_size"], shuffle=True)
        stats = {}

        self.actor_optimizer.zero_grad()
        for batch in transitions:
            batch = batch.to(self.device)
            loss = torch.tensor([0.0], device=self.device)
            states = (batch.constraint_features, batch.edge_index, batch.edge_attr,batch.variable_features,batch.action_set,batch.action_set_size)
            actions = batch.action_idx.unsqueeze(1)
            actor_loss ,entropy = self.calc_policy_loss(states, actions)
            actor_loss /= n_samples
            loss += actor_loss
            entropy /= n_samples
            loss += - self.config['entropy_bonus']*entropy

            loss.backward()
            clip_grad_norm_(self.actor_local.parameters(), self.clip_grad_param)

            # Update stats
            stats['actor_loss'] = stats.get('actor_loss', 0.0) + actor_loss.item()
            stats['loss'] = stats.get('loss', 0.0) + loss.item()
            stats['entropy'] = stats.get('entropy', 0.0) + entropy.item()
        self.actor_optimizer.step()

        self.critic1_optimizer.zero_grad()
        for batch in transitions:
            batch = batch.to(self.device)
            loss = torch.tensor([0.0], device=self.device)
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
            critic1_loss = self.calc_q_loss(states, actions, rewards, dones, next_states)
            critic1_loss /= n_samples
            critic1_loss.backward()
            clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)

            # Update stats
            stats['critic1_loss'] = stats.get('critic1_loss', 0.0) + critic1_loss.item()
            stats['critic2_loss'] = stats.get('critic2_loss', 0.0) 

        self.critic1_optimizer.step()
        return stats


    def learn(self, batch):
        self.step += 1
        states = (batch.constraint_features, batch.edge_index, batch.edge_attr, 
                    batch.variable_features,batch.action_set,batch.action_set_size)
        actions = batch.action_idx.unsqueeze(1)
        rewards = batch.reward
        next_states = (batch.constraint_features_n, batch.edge_index_n, batch.edge_attr_n, 
                        batch.variable_features_n,batch.action_set_n,batch.action_set_n_size)
        dones = batch.done

        actor_loss,_ = self.calc_policy_loss(states, actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        critic1_loss, critic2_loss = self.calc_q_loss(states, actions, rewards, dones, next_states)

        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()

        if self.step % self.hard_update_every == 0:
            # ----------------------- update target networks ----------------------- #
            self.hard_update(self.critic1, self.critic1_target)
            self.hard_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), critic1_loss.item(), critic2_loss.item(), 0
    
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

def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def save(config, model, wandb, stat="offline",ep=None):
    import os
    models_dir = os.path.join(config['model_dir'],"models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    if not ep == None:
        torch.save(model.actor_local.state_dict(), models_dir +'/actor_'+ stat + str(ep) + ".pth")
        # wandb.save(models_dir +'/actor_'+ str(ep) + ".pth")
        torch.save(model.critic1.state_dict(), models_dir +'/critic1_'+ stat + str(ep) + ".pth")
        # wandb.save(models_dir +'/critic1_'+ str(ep) + ".pth")
        torch.save(model.value_net.state_dict(), models_dir +'/value_'+ stat + str(ep) + ".pth")
        # wandb.save(models_dir +'/critic1_'+ str(ep) + ".pth")
    else:
        torch.save(model.actor_local.state_dict(), models_dir +'/actor_'+ stat + "best.pth")
        # wandb.save(models_dir +'/actor_'+ "best.pth")
        torch.save(model.critic1.state_dict(), models_dir +'/critic1_'+ stat + "best.pth")
        # wandb.save(models_dir +'/critic1_'+ "best.pth")
        torch.save(model.value_net.state_dict(), models_dir +'/value_' +stat+ "best.pth")
        # wandb.save(models_dir +'/value_'+ "best.pth")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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