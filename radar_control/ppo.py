"""Clipped PPO with duration-aware GAE for the trajectory control environment."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .dqn import QNetwork


class PPO:
    def __init__(self, dimension=344, seed=7, action_count=2):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.action_count = action_count
        self.actor = QNetwork(dimension, action_count)
        self.critic = nn.Sequential(nn.Linear(dimension, 128), nn.Tanh(), nn.Linear(128, 64),
                                    nn.Tanh(), nn.Linear(64, 1))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=.00005)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=.0003)

    def action(self, state, sample=False):
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(state).unsqueeze(0))
            if not sample:
                if self.action_count == 2:
                    logits[0, 1] += getattr(self, 'switch_bias', 0.)
                return int(logits.argmax(1).item())
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            value = self.critic(torch.as_tensor(state).unsqueeze(0)).item()
            return int(action.item()), float(distribution.log_prob(action).item()), float(value)

    def update(self, states, actions, rewards, durations, log_probs, values):
        # The final transition reaches the finite episode boundary, so bootstrap is 0.
        rewards, values = np.array(rewards), np.array(values)
        discounts = .99 ** (np.array(durations) / 5)
        lambdas = .95 ** (np.array(durations) / 5)
        advantages = np.empty(len(rewards), dtype=np.float32)
        accumulator, next_value = 0., 0.
        for i in range(len(rewards) - 1, -1, -1):
            delta = rewards[i] + discounts[i] * next_value - values[i]
            accumulator = delta + discounts[i] * lambdas[i] * accumulator
            advantages[i] = accumulator
            next_value = values[i]
        returns = torch.as_tensor(advantages + values, dtype=torch.float32)
        advantage = torch.as_tensor((advantages - advantages.mean()) / (advantages.std() + 1e-8))
        states = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)
        old_logs = torch.as_tensor(log_probs)
        last_kl = 0.
        for epoch in range(8):
            indexes = self.rng.permutation(len(states))
            for start in range(0, len(states), 128):
                idx = indexes[start:start + 128]
                distribution = Categorical(logits=self.actor(states[idx]))
                new_logs = distribution.log_prob(actions[idx])
                ratio = (new_logs - old_logs[idx]).exp()
                unclipped = ratio * advantage[idx]
                clipped = ratio.clamp(.8, 1.2) * advantage[idx]
                # Low-weight expert regularization stabilizes training; expert is disclosed.
                expert = (states[idx, 330] <= 0).long()
                if self.action_count == 4:
                    phase = states[idx, 288:292].argmax(1)
                    expert = (phase + expert) % 4
                imitation = nn.functional.cross_entropy(self.actor(states[idx]), expert)
                actor_loss = -torch.minimum(unclipped, clipped).mean() - .005 * distribution.entropy().mean() + .01 * imitation
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
                self.actor_optimizer.step()
                critic_loss = nn.functional.smooth_l1_loss(self.critic(states[idx]).squeeze(1), returns[idx])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.critic_optimizer.step()
            with torch.no_grad():
                new_logs = Categorical(logits=self.actor(states)).log_prob(actions)
                last_kl = float(((new_logs - old_logs).exp() - 1 - (new_logs - old_logs)).mean())
            if last_kl > .02:
                break
        return dict(actor_loss=float(actor_loss.item()), critic_loss=float(critic_loss.item()), approx_kl=last_kl)

    def save(self, path, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), metadata=metadata), path)

    def load(self, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])
        self.switch_bias = data['metadata'].get('switch_bias', 0.)
        return data['metadata']
