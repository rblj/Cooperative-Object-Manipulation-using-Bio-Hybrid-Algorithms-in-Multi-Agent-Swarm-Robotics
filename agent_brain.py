import torch
import torch.nn as nn
import numpy as np
from swarm_env import SwarmEnvironment

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. MUST DEFINE THE BRAIN STRUCTURE (ACTOR) ---
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        return self.net(obs), torch.exp(self.log_std)

# --- 2. EXECUTION LOGIC ---
def run_expert_brain():
    env = SwarmEnvironment(num_agents=5)
    obs, info = env.reset()
    
    # Initialize the brain
    actor = Actor(obs_dim=30, action_dim=2).to(device)
    
    # LOAD THE PEAK CHECKPOINT (60k was your best result)
    model_path = "models/bio_actor_60000.pt" 
    print(f"Loading weights from {model_path}...")
    
    try:
        actor.load_state_dict(torch.load(model_path))
        actor.eval()
        print("Brain Loaded Successfully. Press PLAY in Webots!")
    except FileNotFoundError:
        print(f"Error: {model_path} not found! Check your models folder.")
        return

    for t in range(1000): # Run for a long time to allow box pushing
        obs_tensor = torch.FloatTensor(obs).to(device)
        with torch.no_grad():
            mu, _ = actor(obs_tensor)
            actions = mu.cpu().numpy()
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if terminated:
            print(f"SUCCESS! Box reached goal in {t} steps.")
            break
        if truncated:
            print("Time limit reached.")
            break

if __name__ == "__main__":
    run_expert_brain()