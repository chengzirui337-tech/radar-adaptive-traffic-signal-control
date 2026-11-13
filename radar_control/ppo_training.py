"""Train an expert-initialized PPO policy and select only on validation traffic."""
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .config import Config
from .decision_env import DecisionEnv
from .experiment import prepare
from .optimization import VALIDATION_SEEDS, FINAL_TEST_SEEDS, run_policy
from .ppo import PPO
from .predictive_training import evaluate_predictive
from .scenario import generate_demand
from .signals import ActuatedController


def train_ppo(root, episodes=100, model_seed=7, phase_actions=False):
    config = replace(Config(), decision_seconds=1)
    root.mkdir(parents=True, exist_ok=True)
    prepare(root, config, VALIDATION_SEEDS)
    protocol = dict(method=('phase-selective ' if phase_actions else '') + 'trajectory-predictive expert-initialized semi-Markov PPO',
                    dimension=344, model_seed=model_seed, rl_episodes=episodes,
                    action_count=4 if phase_actions else 2,
                    starvation_guard='if observed continuous wait >=90 s, serve oldest waiting phase' if phase_actions else 'cyclic phases',
                    expert_demand_seeds=list(range(30001, 30005)),
                    train_demand_seeds=list(range(60001 if phase_actions else 50001,
                                                 (60001 if phase_actions else 50001) + episodes)),
                    validation_demand_seeds=list(VALIDATION_SEEDS), final_test_demand_seeds=list(FINAL_TEST_SEEDS),
                    expert='unchanged 40 m actuated rule; behavior cloning and 0.01 cross-entropy regularizer',
                    timing='1 s holds; yellow/all-red/minimum-green enforced without learned decisions',
                    reward='sum of instantaneous speed-loss plus 0.001 accumulated wait, integrated over time / 50',
                    gamma_per_5s=.99, gae_lambda_per_5s=.95, clip=.2, actor_learning_rate=.00005,
                    critic_learning_rate=.0003, entropy_weight=.005, update_epochs=8, target_kl=.02,
                    baseline_unchanged=True, final_test_used_for_selection=False,
                    checkpoint_selection='minimum mean delay across fixed validation seeds')
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    baseline = [run_policy(root, config, s, 'actuated') for s in VALIDATION_SEEDS]
    pd.DataFrame(baseline).to_csv(root / 'validation_actuated.csv', index=False)
    agent = PPO(344, model_seed, action_count=protocol['action_count'])
    teacher = ActuatedController()
    examples, labels, targets = [], [], []
    for seed in protocol['expert_demand_seeds']:
        route = generate_demand(root / 'data' / f'seed_{seed}', config, seed)
        episode_rewards, episode_durations = [], []
        with DecisionEnv(root / 'network', route, root / 'expert' / f'seed_{seed}',
                         config, seed, backend='libsumo') as env:
            state = env.initial_state()
            while env.time < config.horizon:
                action = teacher.action(env)
                if phase_actions:
                    action = (env.signal.phase + action) % 4
                step = env.phase_step if phase_actions else env.decision_step
                nxt, reward, done, duration = step(action)
                examples.append(state)
                labels.append(action)
                episode_rewards.append(reward)
                episode_durations.append(duration)
                state = nxt
        values, value = [], 0.
        for reward, duration in zip(episode_rewards[::-1], episode_durations[::-1]):
            value = reward + .99 ** (duration / 5) * value
            values.append(value)
        targets.extend(values[::-1])
        print(f'PPO EXPERT seed={seed} decisions={len(examples)}', flush=True)
    x = torch.as_tensor(np.array(examples), dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    target = torch.as_tensor(targets, dtype=torch.float32)
    cloning_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=.0003)
    for update in range(1800):
        idx = agent.rng.integers(len(x), size=256)
        loss = torch.nn.functional.cross_entropy(agent.actor(x[idx]), y[idx])
        cloning_optimizer.zero_grad()
        loss.backward()
        cloning_optimizer.step()
        vloss = torch.nn.functional.smooth_l1_loss(agent.critic(x[idx]).squeeze(1), target[idx])
        agent.critic_optimizer.zero_grad()
        vloss.backward()
        agent.critic_optimizer.step()
    with torch.no_grad():
        accuracy = float(agent.actor(x).argmax(1).eq(y).float().mean())
        # Soften only confidence, preserving greedy expert-initialized decisions.
        agent.actor.layers[-1].weight.mul_(.3)
        agent.actor.layers[-1].bias.mul_(.3)
    (root / 'pretraining.json').write_text(json.dumps(dict(examples=len(x), gradient_updates=1800,
        training_accuracy=accuracy, critic_target='duration-discounted expert Monte Carlo return',
        greedy_policy_preserved=True), indent=2), encoding='utf-8')
    history, validations = [], []
    best = float('inf')

    def validate(episode):
        nonlocal best
        rows = evaluate_predictive(root, config, agent, VALIDATION_SEEDS, f'episode_{episode:03d}')
        validations.extend(dict(episode=episode, **r) for r in rows)
        pd.DataFrame(validations).to_csv(root / 'validation_history.csv', index=False)
        mean = float(np.mean([r['mean_delay_s'] for r in rows]))
        metadata = dict(episodes=episode, dimension=344, model_seed=model_seed,
                        action_count=protocol['action_count'],
                        config=config.to_dict(), demand_seeds=protocol['train_demand_seeds'][:episode],
                        expert_seeds=protocol['expert_demand_seeds'], validation_seeds=list(VALIDATION_SEEDS),
                        validation_mean_delay_s=mean, method=protocol['method'], observation='full',
                        checkpoint_selection=protocol['checkpoint_selection'])
        agent.save(root / 'models' / f'episode_{episode:03d}.pt', metadata)
        if mean < best:
            best = mean
            agent.save(root / 'models' / 'best.pt', metadata)
        print(f'PPO VALIDATION episode={episode} delay={mean:.3f}s best={best:.3f}s', flush=True)

    validate(0)
    for episode in range(1, episodes + 1):
        started = time.perf_counter()
        seed = (60000 if phase_actions else 50000) + episode
        route = generate_demand(root / 'data' / f'seed_{seed}', config, seed)
        states, actions, rewards, durations, logs, values = [], [], [], [], [], []
        with DecisionEnv(root / 'network', route, root / 'training' / f'episode_{episode:03d}',
                         config, seed, backend='libsumo') as env:
            state = env.initial_state()
            while env.time < config.horizon:
                action, logprob, value = agent.action(state, sample=True)
                step = env.phase_step if phase_actions else env.decision_step
                nxt, reward, done, duration = step(action)
                states.append(state)
                actions.append(action)
                logs.append(logprob)
                values.append(value)
                rewards.append(reward)
                durations.append(duration)
                state = nxt
        losses = agent.update(states, actions, rewards, durations, logs, values)
        history.append(dict(episode=episode, demand_seed=seed, return_sum=float(np.sum(rewards)),
                            decisions=len(states), wall_seconds=time.perf_counter() - started, **losses))
        pd.DataFrame(history).to_csv(root / 'training_history.csv', index=False)
        print(f'PPO episode={episode}/{episodes} return={sum(rewards):.2f} '
              f'seconds={history[-1]["wall_seconds"]:.1f}', flush=True)
        if episode % 10 == 0 or episode == episodes:
            validate(episode)
    (root / 'training_complete.json').write_text(json.dumps(dict(best_validation_mean_delay_s=best,
        rl_episodes=episodes), indent=2), encoding='utf-8')
