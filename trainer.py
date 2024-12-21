import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import deque

from biggerDQN import DQN
import numpy as np
from tqdm import tqdm #Adding tqdm

class ReplayBuffer(Dataset):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

class Trainer:
    def __init__(
        self, 
        env, 
        input_shape=None, 
        action_shape=None, 
        lr=1e-3,
        gamma=0.9,
        epsilon = 1.0,
        epsilon_min = 0.1,
        epsilon_decay = 0.9,
        device= "cuda"
    ):
        self.env = env

        self.q_network = DQN(input_shape, action_shape).to(device)
        self.target_network = DQN(input_shape, action_shape).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Buffer size dự tính được chọn sau các lần chạy trước đó
        self.replay_buffer = ReplayBuffer(buffer_size=50000)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.steplr = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=0.9)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_target_every = 5

    def select_action(self, observation, agent):
        """Select an action using epsilon-greedy strategy."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if np.random.rand() <= self.epsilon:
            return self.env.action_space(agent).sample()

        observation = torch.FloatTensor(observation).unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.inference_mode():
            q_values = self.q_network(observation)
        return torch.argmax(q_values, dim=1).item()

    def train(self, episodes=100, batch_size=512):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        episode_bar = tqdm(range(episodes), desc="Training Episodes")  # TQDM progress bar for episodes
        self.q_network.train()

        for episode in episode_bar:
            self.env.reset()
            
            total_reward = 0
            reward_for_agent = {agent: 0 for agent in self.env.agents if agent.startswith('blue')}
            prev_observation = {}
            prev_action = {}
            step = 0 # Để quan sát số step mỗi episode

            # Vòng lặp cho 1 trận
            for agent in self.env.agent_iter():
                step += 1
                
                observation, reward, termination, truncation, info = self.env.last()
                observation = np.transpose(observation, (2, 0, 1))
                
                agent_handle = agent.split('_')[0]
                
                if agent_handle == 'blue':
                    total_reward += reward
                    reward_for_agent[agent] += reward
                    
                if termination or truncation:
                    action = None
                else:
                    if agent_handle == 'blue':
                        action = self.select_action(observation, agent)
                    else:
                        action = self.env.action_space(agent).sample()
    
                    if agent_handle == 'blue':
                        # Thêm thông tin vào replay buffer với các agent đã action ít nhất 1 lần. 
                        if agent in prev_observation and agent in prev_action:
                            self.replay_buffer.add(
                                prev_observation[agent],
                                prev_action[agent],
                                reward,  
                                observation,
                                termination
                            )

                        prev_observation[agent] = observation
                        prev_action[agent] = action
    
                self.env.step(action)
            
            dataloader = DataLoader(self.replay_buffer, batch_size=batch_size, shuffle=True, drop_last=True)

            # Update trọng số mạng.
            for states, actions, rewards, next_states, dones in dataloader:
                states = states.to(dtype=torch.float32, device=device)
                actions = actions.to(dtype=torch.long, device=device)
                rewards = rewards.to(dtype=torch.float32, device=device)
                next_states = next_states.to(dtype=torch.float32, device=device)
                dones = dones.to(dtype=torch.float32, device=device)

                current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.inference_mode():
                    next_q_values = self.target_network(next_states).max(1)[0]
                expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

                loss = self.criterion(current_q_values, expected_q_values)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Cập nhật trọng số cho target_network mỗi 5 episode
            if (episode + 1) % self.update_target_every == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.steplr.step()
    
            tqdm.write(f"Episode {episode}: Epsilon={self.epsilon:.2f}, Total Reward={total_reward:.2f}, "
               f"Steps={step}")
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        