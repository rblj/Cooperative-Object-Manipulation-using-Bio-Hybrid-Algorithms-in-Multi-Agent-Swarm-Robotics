import torch
import numpy as np
import pandas as pd
from swarm_env import SwarmEnvironment
from mappo_algorithm import BioHybridMAPPO 
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def run_final_thesis_train():
    total_timesteps = 100000
    env = SwarmEnvironment(num_agents=5)
    model = BioHybridMAPPO(num_agents=5, obs_dim=30, state_dim=26)
    writer = SummaryWriter(log_dir="runs/BIO_FINAL")
    
    csv_data, curr_step = [], 0
    pbar = tqdm(total=total_timesteps, desc="[TRAINING]")

    while curr_step < total_timesteps:
        obs, info = env.reset()
        state = info.get("state")
        ep_active, ep_rewards = [], []
        
        for t in range(300):
            actions, log_probs, values = model.select_actions(obs, state)
            next_obs, rewards, term, trunc, info = env.step(actions)
            
            model.store_transition(obs, actions, np.array(rewards, dtype=np.float32), 
                                  term or trunc, values, log_probs, state)
            
            curr_step += 1
            pbar.update(1)
            
            ep_active.append(float(info.get("active", 0)))
            ep_rewards.append(np.mean(rewards))

            if curr_step % 100 == 0:
                writer.add_scalar("Step/Reward", np.mean(rewards), curr_step)
            
            if len(model.buffer) >= 1200: model.update()
            if term or trunc:
                csv_data.append({"step": curr_step, "success": 1 if term else 0, "dist": info.get("dist")})
                break
            obs, state = next_obs, info.get("state")

        if len(csv_data) % 5 == 0:
            pd.DataFrame(csv_data).to_csv("results.csv", index=False)

    model.save_model(tag="FINAL_POLISHED")
    print("\n[SUCCESS] Training Complete.")

if __name__ == "__main__":
    run_final_thesis_train()