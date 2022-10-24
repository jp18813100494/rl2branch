import torch
import torch_geometric
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from iql.network import Critic, Actor, Value


class IQL(nn.Module):
    def __init__(self,
                state_size,
                action_size,
                config,
                actor_lr,
                critic_lr,
                hidden_size,
                tau,
                gamma,
                temperature,
                expectile,
                hard_update_every,
                clip_grad_param,
                seed,
                device
                ): 
        super(IQL, self).__init__()
        self.config = config
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = torch.FloatTensor([gamma]).to(device)
        self.hard_update_every = hard_update_every
        self.clip_grad_param = clip_grad_param
        self.temperature = torch.FloatTensor([temperature]).to(device)
        self.expectile = torch.FloatTensor([expectile]).to(device)
        self.seed=seed
           
        # Actor Network 
        self.actor_local = Actor(hidden_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)     
        
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(hidden_size, 2).to(device)
        self.critic2 = Critic(hidden_size, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr) 
        
        self.value_net = Value(hidden_size).to(device)
        
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=critic_lr)
        self.step = 0

    
    def sample_action_idx(self, states, greedy):
        #TODO； revise for efficient evaluation
        if isinstance(greedy, bool):
            greedy = torch.tensor(np.repeat(greedy, len(states), dtype=torch.long))
        elif not isinstance(greedy, torch.Tensor):
            greedy = torch.tensor(greedy, dtype=torch.long)

        states_loader = torch_geometric.data.DataLoader(states, batch_size=self.config['batch_size'])
        greedy_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(greedy), batch_size=self.config['batch_size'])
        eval = greedy
        action_idxs = []
        for batch, (greedy,) in zip(states_loader, greedy_loader):
            with torch.no_grad():
                batch = batch.to(self.device)
                
                logits = self.actor_local(batch, eval)
                # logits = logits[batch.action_set]

                logits_end = batch.action_set_size.cumsum(-1)
                logits_start = logits_end - batch.action_set_size
                for start, end, greedy in zip(logits_start, logits_end, greedy):
                    if greedy:
                        action_idx = logits[start:end].argmax()
                    else:
                        action_idx = torch.distributions.categorical.Categorical(logits=logits[start:end]).sample()
                    action_idxs.append(action_idx.item())

        return action_idxs

    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = state.to(self.device)
        with torch.no_grad():
            action = self.actor_local.get_action(state, eval)
        return action

    def calc_policy_loss(self, states, actions):
        with torch.no_grad():
            v = self.value_net(states)
            q1 = self.critic1_target(states).gather(1, actions.long())
            q2 = self.critic2_target(states).gather(1, actions.long())
            min_Q = torch.min(q1,q2)

        exp_a = torch.exp((min_Q - v) * self.temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(self.device)).squeeze(-1)

        _, dist = self.actor_local.evaluate(states)
        log_probs = dist.log_prob(actions.squeeze(-1))
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss
    
    def calc_value_loss(self, states, actions):
        with torch.no_grad():
            q1 = self.critic1_target(states).gather(1, actions.long())
            q2 = self.critic2_target(states).gather(1, actions.long())
            min_Q = torch.min(q1,q2)
        
        value = self.value_net(states)
        value_loss = loss(min_Q - value, self.expectile).mean()
        return value_loss
    
    def calc_q_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.gamma * (1 - dones) * next_v) 

        q1 = self.critic1(states).gather(1, actions.long())
        q2 = self.critic2(states).gather(1, actions.long())
        critic1_loss = ((q1 - q_target)**2).mean() 
        critic2_loss = ((q2 - q_target)**2).mean()
        return critic1_loss, critic2_loss


    def learn(self, batch):
        self.step += 1
        states = (batch.constraint_features, batch.edge_index, batch.edge_attr, 
                    batch.variable_features,batch.action_set,batch.action_set_size)
        actions = batch.action_idx.unsqueeze(1)
        rewards = batch.reward
        next_states = (batch.constraint_features_n, batch.edge_index_n, batch.edge_attr_n, 
                        batch.variable_features_n,batch.action_set_n,batch.action_set_n_size)
        dones = batch.done

        assert batch.edge_index.max()<batch.variable_features.shape[0]
        assert batch.edge_index_n.max()<batch.variable_features_n.shape[0]
        
        # states, actions, rewards, next_states, dones = batch        
        self.value_optimizer.zero_grad()
        value_loss = self.calc_value_loss(states, actions)
        value_loss.backward()
        self.value_optimizer.step()

        actor_loss = self.calc_policy_loss(states, actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        critic1_loss, critic2_loss = self.calc_q_loss(states, actions, rewards, dones, next_states)

        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        if self.step % self.hard_update_every == 0:
            # ----------------------- update target networks ----------------------- #
            self.hard_update(self.critic1, self.critic1_target)
            self.hard_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), critic1_loss.item(), critic2_loss.item(), value_loss.item()
    
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
    if not os.path.exists(config["model_dir"]):
        os.makedirs(config["model_dir"], exist_ok=True)
    if not ep == None:
        torch.save(model.actor_local.state_dict(), config["model_dir"] +'/actor_'+ stat + str(ep) + ".pth")
        wandb.save(config["model_dir"] +'/actor_'+ str(ep) + ".pth")
        torch.save(model.critic1.state_dict(), config["model_dir"] +'/critic1_'+ stat + str(ep) + ".pth")
        wandb.save(config["model_dir"] +'/critic1_'+ str(ep) + ".pth")
        torch.save(model.value_net.state_dict(), config["model_dir"] +'/value_'+ stat + str(ep) + ".pth")
        wandb.save(config["model_dir"] +'/critic1_'+ str(ep) + ".pth")
    else:
        torch.save(model.actor_local.state_dict(), config["model_dir"] +'/actor_'+ stat + "best.pth")
        wandb.save(config["model_dir"] +'/actor_'+ "best.pth")
        torch.save(model.critic1.state_dict(), config["model_dir"] +'/critic1_'+ stat + "best.pth")
        wandb.save(config["model_dir"] +'/critic1_'+ "best.pth")
        torch.save(model.value_net.state_dict(), config["model_dir"] +'/value_' +stat+ "best.pth")
        wandb.save(config["model_dir"] +'/value_'+ "best.pth")