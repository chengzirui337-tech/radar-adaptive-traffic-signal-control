"""Small Double DQN; the evaluation policy is greedy and never updated."""
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)


class QNetwork(nn.Module):
    def __init__(self, dimension, actions=2):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, actions))

    def forward(self, state):
        return self.layers(state)


class DQN:
    def __init__(self, config, seed=0, dimension=None):
        torch.manual_seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.config = config
        dimension = config.state_dim if dimension is None else dimension
        self.online, self.target = QNetwork(dimension), QNetwork(dimension)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        n, d = config.replay_size, dimension
        self.states = np.empty((n, d), dtype=np.float32)
        self.next_states = np.empty((n, d), dtype=np.float32)
        self.actions = np.empty(n, dtype=np.int64)
        self.rewards = np.empty(n, dtype=np.float32)
        self.dones = np.empty(n, dtype=np.float32)
        self.discounts = np.empty(n, dtype=np.float32)
        self.count = self.steps = self.updates = 0

    def action(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(2))
        with torch.no_grad():
            return int(self.online(torch.as_tensor(state).unsqueeze(0)).argmax(1).item())

    def remember(self, state, action, reward, next_state, done, discount=None):
        i = self.steps % self.config.replay_size
        self.states[i], self.next_states[i] = state, next_state
        self.actions[i], self.rewards[i], self.dones[i] = action, reward, done
        self.discounts[i] = self.config.gamma if discount is None else discount
        self.count = min(self.count + 1, self.config.replay_size)
        self.steps += 1

    def learn(self):
        c = self.config
        if self.count < c.warmup_steps or self.steps % c.train_every:
            return None
        idx = self.rng.integers(self.count, size=c.batch_size)
        state, nxt = torch.from_numpy(self.states[idx]), torch.from_numpy(self.next_states[idx])
        actions = torch.from_numpy(self.actions[idx]).unsqueeze(1)
        rewards, dones = torch.from_numpy(self.rewards[idx]), torch.from_numpy(self.dones[idx])
        prediction = self.online(state).gather(1, actions).squeeze(1)
        with torch.no_grad():
            best = self.online(nxt).argmax(1, keepdim=True)
            discounts = torch.from_numpy(self.discounts[idx])
            target = rewards + discounts * (1 - dones) * self.target(nxt).gather(1, best).squeeze(1)
        loss = nn.functional.smooth_l1_loss(prediction, target)
        if getattr(self, 'expert_feature_index', None) is not None:
            # Optional transparent imitation regularizer for the improved variant.
            expert = (state[:, self.expert_feature_index] <= 0).long()
            loss = loss + self.imitation_weight * nn.functional.cross_entropy(self.online(state), expert)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % c.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path, metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': self.online.state_dict(), 'metadata': metadata}, path)

    def load(self, path):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        self.online.load_state_dict(payload['state_dict'])
        self.target.load_state_dict(self.online.state_dict())
        self.online.eval()
        return payload['metadata']
