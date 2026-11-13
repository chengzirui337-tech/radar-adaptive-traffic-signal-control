"""Expert-initialized semi-Markov Double DQN, with validation-only selection."""
from dataclasses import replace
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

from .config import Config
from .decision_env import DecisionEnv
from .dqn import DQN
from .experiment import prepare
from .optimization import VALIDATION_SEEDS, FINAL_TEST_SEEDS, run_policy
from .scenario import generate_demand
from .signals import ActuatedController


def evaluate_predictive(root, config, agent, seeds, label, record=False):
    rows = []
    for seed in seeds:
        folder = root / 'data' / f'seed_{seed}'
        route = folder / 'demand.rou.xml'
        if not route.exists():
            route = generate_demand(folder, config, seed)
        output = root / ('evaluation' if record else 'validation') / label / f'seed_{seed}'
        with DecisionEnv(root / 'network', route, output, config, seed,
                         backend='libsumo', record=record, export_radar=(record and seed == seeds[0])) as env:
            state = env.initial_state()
            while env.time < config.horizon:
                step = env.phase_step if getattr(agent, 'action_count', 2) == 4 else env.decision_step
                state, _, _, _ = step(agent.action(state))
            while env.has_pending() and env.time < config.horizon + config.drain_seconds:
                state, _, _, _ = step(agent.action(state), config.horizon + config.drain_seconds)
        trips = ET.parse(output / 'tripinfo.xml').getroot().findall('tripinfo')
        if any(float(t.get('arrival')) < 0 for t in trips):
            raise ValueError('Unfinished trips in predictive evaluation')
        rows.append(dict(method=label, seed=int(seed), mean_delay_s=float(np.mean([
            float(t.get('timeLoss')) + float(t.get('departDelay')) for t in trips])),
                         mean_stops=float(np.mean([float(t.get('waitingCount')) for t in trips])),
                         vehicles=len(trips), collisions=env.collisions, teleports=env.teleports))
    return rows


def train_predictive(root, episodes=100, model_seed=7):
    config = replace(Config(), decision_seconds=1, learning_rate=.0001, batch_size=128,
                     target_update_steps=500, train_every=2)
    root.mkdir(parents=True, exist_ok=True)
    prepare(root, config, VALIDATION_SEEDS)
    protocol = dict(method='trajectory-predictive expert-initialized semi-Markov Double DQN',
                    dimension=DecisionEnv.feature_dimension, model_seed=model_seed,
                    rl_episodes=episodes, expert_demand_seeds=list(range(30001, 30005)),
                    train_demand_seeds=list(range(40001, 40001 + episodes)),
                    validation_demand_seeds=list(VALIDATION_SEEDS), final_test_demand_seeds=list(FINAL_TEST_SEEDS),
                    expert='unchanged 40 m actuated rule, used only for initialization and 0.02 imitation regularizer',
                    reward='integral of -(sum(1-v/13.89) + 0.001*cumulative_wait)/10, scaled by seconds/5',
                    timing='1 second discretionary holds; forced yellow/all-red/min-green periods advance without actions',
                    discount='0.99 ** (actual transition duration seconds / 5)',
                    state='original 296 trajectory bins + 32 phase arrivals/queue features + current/next 8 each',
                    baseline_unchanged=True, final_test_used_for_selection=False,
                    checkpoint_selection='minimum mean delay across three fixed validation seeds')
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    baseline = [run_policy(root, config, s, 'actuated') for s in VALIDATION_SEEDS]
    pd.DataFrame(baseline).to_csv(root / 'validation_actuated.csv', index=False)
    agent = DQN(config, model_seed, dimension=DecisionEnv.feature_dimension)
    expert = ActuatedController()
    examples, labels = [], []
    for seed in protocol['expert_demand_seeds']:
        route = generate_demand(root / 'data' / f'seed_{seed}', config, seed)
        with DecisionEnv(root / 'network', route, root / 'expert' / f'seed_{seed}',
                         config, seed, backend='libsumo') as env:
            state = env.initial_state()
            while env.time < config.horizon:
                action = expert.action(env)
                nxt, reward, done, duration = env.decision_step(action)
                examples.append(state)
                labels.append(action)
                agent.remember(state, action, reward, nxt, done, config.gamma ** (duration / 5))
                state = nxt
        print(f'EXPERT seed={seed} decision_examples={len(examples)}', flush=True)
    x = torch.from_numpy(np.array(examples))
    y = torch.as_tensor(labels, dtype=torch.long)
    # Actual supervised pretraining on trajectories from training-only traffic.
    for update in range(1600):
        idx = agent.rng.integers(len(x), size=256)
        loss = torch.nn.functional.cross_entropy(agent.online(x[idx]), y[idx])
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
    with torch.no_grad():
        accuracy = float(agent.online(x).argmax(1).eq(y).float().mean())
    agent.target.load_state_dict(agent.online.state_dict())
    agent.expert_feature_index = 330  # current-phase count within 40 m; zero means gap-out
    agent.imitation_weight = .02
    (root / 'pretraining.json').write_text(json.dumps(dict(examples=len(x), updates=1600,
        training_accuracy=accuracy, label='training accuracy, not held-out control performance'), indent=2), encoding='utf-8')
    print(f'IMITATION training accuracy={accuracy:.5f}', flush=True)
    validations, history = [], []
    best = float('inf')

    def validate(episode):
        nonlocal best
        rows = evaluate_predictive(root, config, agent, VALIDATION_SEEDS, f'episode_{episode:03d}')
        validations.extend(dict(episode=episode, **r) for r in rows)
        pd.DataFrame(validations).to_csv(root / 'validation_history.csv', index=False)
        mean = float(np.mean([r['mean_delay_s'] for r in rows]))
        metadata = dict(episodes=episode, dimension=DecisionEnv.feature_dimension, model_seed=model_seed,
                        config=config.to_dict(), demand_seeds=protocol['train_demand_seeds'][:episode],
                        expert_seeds=protocol['expert_demand_seeds'], validation_seeds=list(VALIDATION_SEEDS),
                        validation_mean_delay_s=mean, method=protocol['method'], observation='full',
                        checkpoint_selection=protocol['checkpoint_selection'])
        agent.save(root / 'models' / f'episode_{episode:03d}.pt', metadata)
        if mean < best:
            best = mean
            agent.save(root / 'models' / 'best.pt', metadata)
        print(f'PREDICTIVE VALIDATION episode={episode} delay={mean:.3f}s best={best:.3f}s', flush=True)

    validate(0)
    for episode in range(1, episodes + 1):
        started = time.perf_counter()
        seed = 40000 + episode
        route = generate_demand(root / 'data' / f'seed_{seed}', config, seed)
        epsilon = max(.01, .10 - .09 * episode / max(1, episodes * .75))
        total, losses = 0., []
        with DecisionEnv(root / 'network', route, root / 'training' / f'episode_{episode:03d}',
                         config, seed, backend='libsumo') as env:
            state = env.initial_state()
            while env.time < config.horizon:
                action = agent.action(state, epsilon)
                nxt, reward, done, duration = env.decision_step(action)
                agent.remember(state, action, reward, nxt, done, config.gamma ** (duration / 5))
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                total += reward
                state = nxt
        history.append(dict(episode=episode, demand_seed=seed, return_sum=total, epsilon=epsilon,
                            mean_loss=float(np.mean(losses)), wall_seconds=time.perf_counter() - started))
        pd.DataFrame(history).to_csv(root / 'training_history.csv', index=False)
        print(f'PREDICTIVE episode={episode}/{episodes} return={total:.2f} '
              f'seconds={history[-1]["wall_seconds"]:.1f}', flush=True)
        if episode % 10 == 0 or episode == episodes:
            validate(episode)
    (root / 'training_complete.json').write_text(json.dumps(dict(best_validation_mean_delay_s=best,
        rl_episodes=episodes), indent=2), encoding='utf-8')
