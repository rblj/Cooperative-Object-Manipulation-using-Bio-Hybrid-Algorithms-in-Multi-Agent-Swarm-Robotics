import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU()
        )
        self.mu = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        feat = self.base(x)
        return torch.tanh(self.mu(feat)), torch.exp(self.log_std)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.fc(x)

class BioHybridMAPPO:
    def __init__(self, num_agents, obs_dim, state_dim):
        self.num_agents = num_agents
        self.gamma, self.eps_clip = 0.99, 0.2
        self.actors = [Actor(obs_dim, 2) for _ in range(num_agents)]
        self.critic = Critic(state_dim)
        self.actor_opts = [optim.Adam(a.parameters(), lr=5e-4) for a in self.actors]
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=5e-4)
        self.buffer = []

    def select_actions(self, obs, state):
        actions, log_probs, values = [], [], []
        state_t = torch.FloatTensor(state)
        val = self.critic(state_t).detach()
        for i in range(self.num_agents):
            mu, std = self.actors[i](torch.FloatTensor(obs[i]))
            dist = Normal(mu, std)
            act = dist.sample()
            actions.append(act.numpy())
            log_probs.append(dist.log_prob(act).sum().detach())
            values.append(val)
        return np.array(actions), log_probs, values

    def store_transition(self, obs, act, rew, done, val, logp, sta):
        self.buffer.append((obs, act, rew, done, val, logp, sta))

    def update(self):
        if len(self.buffer) < 400: return # Smaller batch for faster updates
        obs_b, act_b, rew_b, don_b, val_b, log_b, sta_b = zip(*self.buffer)
        
        rewards = torch.FloatTensor(np.array(rew_b)).mean(dim=1)
        masks = torch.FloatTensor(1 - np.array(don_b))
        old_vals = torch.stack([v[0] for v in val_b])
        
        returns, disc_rew = [], 0
        for r, m in zip(reversed(rewards), reversed(masks)):
            disc_rew = r + (self.gamma * disc_rew * m)
            returns.insert(0, disc_rew)
        returns = torch.stack(returns)
        adv = (returns - old_vals).detach()

        for _ in range(10): # High epoch for fast learning
            for i in range(self.num_agents):
                o = torch.stack([torch.FloatTensor(ob[i]) for ob in obs_b])
                a = torch.stack([torch.from_numpy(ac[i]) for ac in act_b])
                lp_old = torch.stack([lp[i] for lp in log_b])
                
                mu, std = self.actors[i](o)
                dist = Normal(mu, std)
                lp_new = dist.log_prob(a).sum(dim=-1)
                
                ratio = torch.exp(lp_new - lp_old)
                loss = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()

                self.actor_opts[i].zero_grad()
                loss.backward(retain_graph=True)
                self.actor_opts[i].step()

            c_loss = nn.MSELoss()(self.critic(torch.stack([torch.FloatTensor(s) for s in sta_b])).squeeze(), returns)
            self.critic_opt.zero_grad()
            c_loss.backward()
            self.critic_opt.step()
        self.buffer = []

    def save_model(self, tag="FINAL"):
        for i, a in enumerate(self.actors): torch.save(a.state_dict(), f"actor_{i}_{tag}.pth")
        torch.save(self.critic.state_dict(), f"critic_{tag}.pth")