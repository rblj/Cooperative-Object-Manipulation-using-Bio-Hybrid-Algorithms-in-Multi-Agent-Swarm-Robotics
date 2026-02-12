import pandas as pd
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

class ThesisLogger:
    def __init__(self, log_dir="logs/thesis_data"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.csv_path = os.path.join(log_dir, "swarm_metrics.csv")
        self.data_buffer = []

    def log_step(self, step, metrics: dict):
        # Metrics should contain: 'success_rate', 'cohesion', 'active_agents', 'box_dist'
        metrics['step'] = step
        self.data_buffer.append(metrics)
        
        # TensorBoard for real-time monitoring
        for k, v in metrics.items():
            if k != 'step':
                self.writer.add_scalar(f"Swarm/{k}", v, step)

        # Periodically save to CSV
        if step % 100 == 0:
            df = pd.DataFrame(self.data_buffer)
            df.to_csv(self.csv_path, mode='a', header=not os.path.exists(self.csv_path), index=False)
            self.data_buffer = []

    @staticmethod
    def calculate_cohesion(agent_positions):
        """PSO Metric: Measure how grouped the swarm is."""
        if len(agent_positions) < 2: return 0
        centroid = np.mean(agent_positions, axis=0)
        distances = [np.linalg.norm(pos - centroid) for pos in agent_positions]
        return np.mean(distances)

    @staticmethod
    def calculate_success_rate(box_pos, goal_pos, threshold=2.0):
        return 1.0 if np.linalg.norm(box_pos - goal_pos) < threshold else 0.0