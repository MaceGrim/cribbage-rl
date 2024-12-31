import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cribbage_rl.cribbage_env.core.cards import Card
from cribbage_rl.cribbage_env.core.environment import CribbageEnv
from ..base.base_agent import BaseAgent


class DQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(BaseAgent):
    def __init__(self, player_id: int, state_dim: int = 114, action_dim: int = 52,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, buffer_size: int = 10000,
                 batch_size: int = 64, target_update: int = 10):
        super().__init__(player_id)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        self.steps = 0

    def _get_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def _get_valid_actions(self, hand: List[Card], phase: str,
                          played_cards: List[Card] = None) -> List[int]:
        if phase == "discard":
            return list(range(len(hand)))
        elif phase == "play":
            play_count = sum(card.value for card in played_cards) if played_cards else 0
            return [
                i for i, card in enumerate(hand)
                if play_count + card.value <= 31
            ]
        return []

    def choose_discard(self, hand: List[Card], is_dealer: bool) -> List[Card]:
        self.hand = hand.copy()
        self.is_dealer = is_dealer
        
        # Create a simple state representation for discarding
        state = np.zeros(114, dtype=np.float32)
        for card in hand:
            card_idx = self._card_to_index(card)
            state[card_idx] = 1
        state[104] = float(is_dealer)  # Crib ownership
        state[107 + 1] = 1  # Discard phase
        
        state_tensor = self._get_state_tensor(state)
        valid_actions = list(range(len(hand)))
        
        if random.random() < self.epsilon:
            indices = random.sample(valid_actions, 2)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                sorted_actions = sorted(
                    valid_actions,
                    key=lambda x: q_values[0][x].item(),
                    reverse=True
                )
                indices = sorted_actions[:2]
        
        discard = [hand[i] for i in indices]
        for card in discard:
            self.hand.remove(card)
        return discard

    def _card_to_index(self, card: Card) -> int:
        suit_offset = list(card.suit.__class__).index(card.suit) * 13
        rank_offset = list(card.rank.__class__).index(card.rank)
        return suit_offset + rank_offset

    def choose_play(self, hand: List[Card], played_cards: List[Card],
                   play_count: int) -> Optional[Card]:
        valid_actions = self._get_valid_actions(hand, "play", played_cards)
        if not valid_actions:
            return None
            
        # Create a simple state representation for playing
        state = np.zeros(114, dtype=np.float32)
        # Encode hand
        for card in hand:
            card_idx = self._card_to_index(card)
            state[card_idx] = 1
        # Encode played cards
        for card in played_cards:
            card_idx = 52 + self._card_to_index(card)
            state[card_idx] = 1
        state[104] = float(self.is_dealer)  # Crib ownership
        state[107 + 2] = 1  # Play phase
        state[113] = play_count / 31  # Play count
        
        state_tensor = self._get_state_tensor(state)
        
        if random.random() < self.epsilon:
            idx = random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                valid_q_values = [(i, q_values[0][i].item()) for i in valid_actions]
                idx = max(valid_q_values, key=lambda x: x[1])[0]
        
        return hand[idx]

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        self.memory.push(state, action, reward, next_state, done)
        
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        self.epsilon = max(self.epsilon_end,
                         self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']