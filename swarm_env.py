import gymnasium as gym
from gymnasium import spaces
import socket
import json
import numpy as np
import select

class SwarmEnvironment(gym.Env):
    def __init__(self, num_agents=5, port=12345):
        super().__init__()
        self.num_agents = num_agents
        self.max_steps = 400
        self.step_count = 0
        self.prev_dist = 24.0
        
        self.robot_positions = np.zeros((num_agents, 2))
        self.box_position = np.array([12.0, 0.0]) 
        self.goal_pos = np.array([-12.0, 0.0]) 

        self.observation_space = spaces.Box(low=-50, high=50, shape=(30,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("localhost", port))
        self.server.listen(num_agents)
        self.connections = []

    def _calculate_reward(self, actions):
        rewards = np.zeros(self.num_agents)
        box_pos, goal_pos = self.box_position, self.goal_pos
        
        # 1. ACO: Pheromone Gradient toward the Goal
        box_to_goal = goal_pos - box_pos
        dist_box_to_goal = np.linalg.norm(box_to_goal)
        unit_goal_vec = box_to_goal / (dist_box_to_goal + 1e-6)
        
        # Pushing Zone: 1.2 units directly behind the box
        push_zone = box_pos - (unit_goal_vec * 1.2)
        dist_to_box = np.linalg.norm(self.robot_positions - box_pos, axis=1)
        
        # 2. BEE SWARM: Global Recruitment (Waggle Dance)
        # Immediate signal once contact is made
        is_recruiting = np.any(dist_to_box < 2.0)

        # 3. PSO: Swarm Centroid (Cohesion)
        swarm_center = np.mean(self.robot_positions, axis=0)

        for i in range(self.num_agents):
            pos = self.robot_positions[i]
            r = 0.0

            # --- CRITICAL: STAY BEHIND THE BOX ---
            if pos[0] < (box_pos[0] - 0.3):
                r -= 10000.0 # Absolute barrier

            # --- COOPERATION: SWARM COHESION ---
            # Penalize robots that wander away from the group
            dist_to_swarm = np.linalg.norm(pos - swarm_center)
            r -= 200.0 * dist_to_swarm

            # --- RECRUITMENT LOGIC ---
            dist_to_zone = np.linalg.norm(pos - push_zone)
            if is_recruiting:
                r -= 600.0 * dist_to_zone # Snap to pusher position
            else:
                r -= 100.0 * dist_to_box[i] # Rapid search

            # --- FORMATION: LATERAL DISCIPLINE ---
            lateral_offset = abs(pos[1] - box_pos[1])
            r -= 500.0 * lateral_offset

            # --- SYNERGY: COLLECTIVE PUSH ---
            if dist_to_box[i] < 2.2 and pos[0] > box_pos[0]:
                # Force alignment reward
                r += 5000.0 * np.dot(actions[i], unit_goal_vec)

            rewards[i] = r

        # ACO Progress Reward
        progress = self.prev_dist - dist_box_to_goal
        if progress > 0.001:
            rewards += (progress * 2000000.0)

        self.prev_dist = dist_box_to_goal
        return rewards.astype(np.float32), {"dist": dist_box_to_goal}

    def step(self, actions):
        self.step_count += 1
        for i, conn in enumerate(self.connections):
            try:
                msg = json.dumps({"action": actions[i].tolist(), "id": i}) + "\n"
                conn.sendall(msg.encode())
                ready = select.select([conn], [], [], 0.01)
                if ready[0]:
                    update = json.loads(conn.recv(1024).decode().split('\n')[0])
                    self.robot_positions[i] = [update['x'], update['y']]
                    if 'box_x' in update:
                        self.box_position = np.array([update['box_x'], update['box_y']])
            except: pass

        obs = self._get_obs()
        reward, metrics = self._calculate_reward(actions)
        terminated = bool(metrics["dist"] < 2.1) 
        truncated = bool(self.step_count >= self.max_steps)
        return obs, reward, terminated, truncated, {"state": self._get_state(), **metrics}

    def _get_obs(self):
        obs_list = []
        for i in range(self.num_agents):
            o = np.zeros(30)
            o[0:2] = self.robot_positions[i] / 25.0
            o[2:4] = (self.box_position - self.robot_positions[i]) / 25.0
            o[4:6] = (self.goal_pos - self.robot_positions[i]) / 25.0
            obs_list.append(o)
        return np.array(obs_list, dtype=np.float32)

    def _get_state(self):
        state = np.zeros(26)
        state[0:10] = self.robot_positions.flatten() / 25.0
        state[10:12] = self.box_position / 25.0
        state[12:14] = self.goal_pos / 25.0
        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count, self.prev_dist = 0, 24.0
        while len(self.connections) < self.num_agents:
            conn, _ = self.server.accept()
            self.connections.append(conn)
        return self._get_obs(), {"state": self._get_state()}