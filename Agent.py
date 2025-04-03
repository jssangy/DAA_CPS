import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.LongTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.LongTensor(rewards).to(self.device)
        next_states = torch.LongTensor(next_states).to(self.device)
        dones = torch.LongTensor(dones).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, state_size, hidden_size, action_size, lr, gamma, memory_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DQN 네트워크 초기화
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 옵티마이저 및 손실 함수
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 경험 리플레이 메모리
        self.memory = ReplayBuffer(memory_size)
        
        # 학습 파라미터
        self.gamma = gamma
        self.action_size = action_size
        
        # 현재 학습 상태 추적
        self.current_loss = 0
        self.current_q_value = 0
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        with torch.no_grad():
            state = torch.LongTensor(state).to(self.device)
            q_values = self.policy_net(state)
            self.current_q_value = q_values.max().item()
            return torch.argmax(q_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # 현재 Q 값 계산
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 다음 상태의 최대 Q 값 계산
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        
        # 타겟 Q 값 계산
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Huber Loss 사용 (MSE 대신)
        self.current_loss = F.smooth_l1_loss(current_q.squeeze(), target_q).item()
        
        # 역전파 및 최적화
        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def get_statistics(self):
        return {
            'current_loss': self.current_loss,
            'current_q_value': self.current_q_value,
            'memory_size': len(self.memory)
        }
    
