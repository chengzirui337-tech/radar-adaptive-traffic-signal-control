"""Keep original results; reserve validation and final-test demand seeds up front."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .config import Config
from .dqn import DQN
from .environment import IntersectionEnv
from .experiment import prepare
from .scenario import generate_demand
from .signals import ActuatedController, FixedController

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SEEDS = (1001, 1002, 1003)
FINAL_TEST_SEEDS = tuple(range(2001, 2011))


def run_policy(root, config, seed, method, agent=None, split='validation', record=False):
    folder = root / 'data' / f'seed_{seed}'
    route = folder / 'demand.rou.xml'
    if not route.exists():
        route = generate_demand(folder, config, seed)
    output = root / split / method / f'seed_{seed}'
    controller = ActuatedController() if method == 'actuated' else FixedController() if method == 'fixed' else None
    with IntersectionEnv(root / 'network', route, output, config, seed,
                         backend='libsumo', record=record) as env:
        state, done = env.state(), False
        while not done:
            action = agent.action(state) if controller is None else 0
            state, _, done = env.step(action, controller)
        while env.has_pending() and env.time < config.horizon + config.drain_seconds:
            action = agent.action(state) if controller is None else 0
            state, _, _ = env.step(action, controller, config.horizon + config.drain_seconds)
    trips = ET.parse(output / 'tripinfo.xml').getroot().findall('tripinfo')
    if any(float(t.get('arrival')) < 0 for t in trips):
        raise ValueError('Unfinished trips invalidate validation means.')
    return dict(method=method, seed=int(seed), mean_delay_s=float(np.mean([
        float(t.get('timeLoss')) + float(t.get('departDelay')) for t in trips])),
                mean_stops=float(np.mean([float(t.get('waitingCount')) for t in trips])),
                vehicles=len(trips), collisions=env.collisions, teleports=env.teleports)


def longer_training(root, extra_episodes=160, validate_every=20):
    config = Config()
    root.mkdir(parents=True, exist_ok=True)
    prepare(root, config, VALIDATION_SEEDS)
    protocol = dict(purpose='additional training with unchanged 5-second action timing and legacy reward',
                    original_checkpoint='results/full/models/dqn_full.pt', original_episodes=40,
                    additional_episodes=extra_episodes,
                    train_demand_seeds=list(range(10001, 10001 + extra_episodes)),
                    validation_demand_seeds=list(VALIDATION_SEEDS), final_test_demand_seeds=list(FINAL_TEST_SEEDS),
                    selection='lowest mean vehicle delay over fixed validation seeds',
                    validation_every=validate_every,
                    note='Warm-start weights; original optimizer and replay were not retained. Fresh Adam and replay.',
                    epsilon='0.05 linearly decreasing to 0.01 in the first 75% of additional episodes',
                    baseline_unchanged=True, final_test_used_for_selection=False)
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    baseline = [run_policy(root, config, s, 'actuated') for s in VALIDATION_SEEDS]
    pd.DataFrame(baseline).to_csv(root / 'validation_actuated.csv', index=False)
    print(f'Actuated validation delay={np.mean([r["mean_delay_s"] for r in baseline]):.3f}s', flush=True)
    agent = DQN(config, 7)
    initial = agent.load(ROOT / 'results/full/models/dqn_full.pt')
    history, validations = [], []
    best = float('inf')

    def validate(episode):
        nonlocal best
        rows = [run_policy(root, config, s, f'episode_{episode:03d}', agent) for s in VALIDATION_SEEDS]
        mean = float(np.mean([r['mean_delay_s'] for r in rows]))
        validations.extend(dict(episode=episode, **r) for r in rows)
        pd.DataFrame(validations).to_csv(root / 'validation_history.csv', index=False)
        metadata = dict(initial, episodes=episode, config=config.to_dict(),
                        demand_seeds=initial['demand_seeds'] + protocol['train_demand_seeds'][:max(0, episode - 40)],
                        validation_seeds=list(VALIDATION_SEEDS), validation_mean_delay_s=mean,
                        checkpoint_selection='minimum validation delay; no final test access')
        agent.save(root / 'models' / f'episode_{episode:03d}.pt', metadata)
        if mean < best:
            best = mean
            agent.save(root / 'models' / 'best.pt', metadata)
        print(f'VALIDATION total_episode={episode} delay={mean:.3f}s best={best:.3f}s', flush=True)

    validate(40)
    for extra in range(1, extra_episodes + 1):
        seed = 10000 + extra
        route = generate_demand(root / 'data' / f'seed_{seed}', config, seed)
        epsilon = max(.01, .05 - .04 * extra / max(1, extra_episodes * .75))
        started = time.perf_counter()
        total, losses = 0., []
        with IntersectionEnv(root / 'network', route, root / 'training' / f'episode_{extra + 40:03d}',
                             config, seed, backend='libsumo') as env:
            state, done = env.state(), False
            while not done:
                action = agent.action(state, epsilon)
                nxt, reward, done = env.step(action)
                agent.remember(state, action, reward, nxt, done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                total += reward
                state = nxt
        row = dict(episode=extra + 40, additional_episode=extra, demand_seed=seed, epsilon=epsilon,
                   return_sum=total, mean_loss=float(np.mean(losses)) if losses else None,
                   wall_seconds=time.perf_counter() - started)
        history.append(row)
        pd.DataFrame(history).to_csv(root / 'training_history.csv', index=False)
        print(f'LONGER total_episode={extra + 40}/{extra_episodes + 40} return={total:.2f} '
              f'epsilon={epsilon:.3f} seconds={row["wall_seconds"]:.1f}', flush=True)
        if extra % validate_every == 0 or extra == extra_episodes:
            validate(extra + 40)
    (root / 'training_complete.json').write_text(json.dumps(dict(best_validation_mean_delay_s=best,
        actuated_validation_mean_delay_s=float(np.mean([r['mean_delay_s'] for r in baseline])),
        additional_episodes=extra_episodes), indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['longer', 'predictive', 'ppo', 'ppo4', 'calibrate', 'finalize', 'report'])
    parser.add_argument('--episodes', type=int, default=160, help='Additional episodes after the original 40')
    parser.add_argument('--output', default='results/optimized/longer')
    parser.add_argument('--model-seed', type=int, default=7)
    args = parser.parse_args()
    if args.command == 'longer':
        longer_training((ROOT / args.output).resolve(), args.episodes)
    elif args.command == 'predictive':
        from .predictive_training import train_predictive
        train_predictive((ROOT / args.output).resolve(), args.episodes)
    elif args.command == 'ppo':
        from .ppo_training import train_ppo
        train_ppo((ROOT / args.output).resolve(), args.episodes, args.model_seed)
    elif args.command == 'ppo4':
        from .ppo_training import train_ppo
        train_ppo((ROOT / args.output).resolve(), args.episodes, args.model_seed, phase_actions=True)
    elif args.command == 'finalize':
        from .final_comparison import finalize
        finalize()
    elif args.command == 'calibrate':
        from .calibration import calibrate
        calibrate(ROOT / 'results/optimized/ppo')
    elif args.command == 'report':
        from .optimization_report import write_optimization_report
        write_optimization_report(ROOT / 'results/optimized/comparison')
