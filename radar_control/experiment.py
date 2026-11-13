import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .dqn import DQN
from .environment import IntersectionEnv
from .metrics import aggregate, summarize
from .scenario import build_network, generate_demand
from .signals import ActuatedController, FixedController

ROOT = Path(__file__).resolve().parents[1]


def prepare(root, config, seeds):
    build_network(root / 'network')
    for seed in seeds:
        generate_demand(root / 'data' / f'seed_{seed}', config, seed)
    (root / 'config.json').write_text(json.dumps(config.to_dict(), indent=2), encoding='utf-8')
    import traci
    versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    torch=torch.__version__, traci=getattr(traci, '__version__', '1.28.0'),
                    synthetic_data=True)
    (root / 'versions.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')


def train(root, config, episodes, observation, backend, model_seed):
    variant = f'dqn_{observation}'
    agent = DQN(config, model_seed)
    history = []
    # Held-out evaluation seeds are never used to fit or select the model.
    for episode in range(1, episodes + 1):
        seed = episode
        routes = generate_demand(root / 'data' / f'seed_{seed}', config, seed)
        epsilon = max(.05, 1.0 - .95 * (episode - 1) / max(1, episodes * .8))
        started = time.perf_counter()
        total, losses = 0., []
        with IntersectionEnv(root / 'network', routes, root / 'training' / variant / f'episode_{episode:03d}',
                             config, seed, observation=observation, backend=backend) as env:
            state = env.state()
            done = False
            while not done:
                action = agent.action(state, epsilon)
                nxt, reward, done = env.step(action)
                agent.remember(state, action, reward, nxt, done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                total += reward
                state = nxt
        history.append(dict(episode=episode, demand_seed=seed, epsilon=epsilon, return_sum=total,
                            mean_loss=float(np.mean(losses)) if losses else None,
                            wall_seconds=time.perf_counter() - started))
        pd.DataFrame(history).to_csv(root / 'training' / f'{variant}_history.csv', index=False)
        print(f'{variant} episode {episode}/{episodes} seed={seed} return={total:.1f} '
              f'epsilon={epsilon:.3f} seconds={history[-1]["wall_seconds"]:.1f}', flush=True)
    metadata = dict(episodes=episodes, demand_seeds=list(range(1, episodes + 1)),
                    model_seed=model_seed, observation=observation, config=config.to_dict(),
                    checkpoint_selection='final training episode; no test-based selection')
    agent.save(root / 'models' / f'{variant}.pt', metadata)


def evaluate(root, config, seeds, backend, ablation=False):
    variants = ['fixed', 'actuated', 'dqn_full'] + (['dqn_near'] if ablation else [])
    agents = {}
    for method in variants:
        if method.startswith('dqn'):
            agent = DQN(config)
            metadata = agent.load(root / 'models' / f'{method}.pt')
            if set(seeds) & set(metadata['demand_seeds']):
                raise ValueError('Training and test demand seeds overlap.')
            if metadata['observation'] != method.removeprefix('dqn_'):
                raise ValueError('Checkpoint observation type does not match the requested method.')
            if metadata['config']['stage_seconds'] != config.stage_seconds:
                raise ValueError('Checkpoint and evaluation use different stage lengths.')
            agents[method] = agent
    rows = []
    for seed in seeds:
        folder = root / 'data' / f'seed_{seed}'
        routes = generate_demand(folder, config, seed)
        for method in variants:
            output = root / 'evaluation' / method / f'seed_{seed}'
            controller = FixedController() if method == 'fixed' else ActuatedController() if method == 'actuated' else None
            observation = 'near' if method == 'dqn_near' else 'full'
            started = time.perf_counter()
            with IntersectionEnv(root / 'network', routes, output, config, seed, observation=observation,
                                 backend=backend, record=True, export_radar=(seed == seeds[0])) as env:
                done, state = False, env.state()
                while not done:
                    action = agents[method].action(state) if controller is None else 0
                    state, _, done = env.step(action, controller)
                # Stop demand at 75 min and clear remaining trips; do not pad main-window queue statistics.
                max_end = config.horizon + config.drain_seconds
                while env.has_pending() and env.time < max_end:
                    action = agents[method].action(state) if controller is None else 0
                    state, _, _ = env.step(action, controller, end_time=max_end)
            measures = summarize(output, folder, config, method, seed)
            rows.extend(measures)
            whole = measures[0]
            print(f'{method} test seed={seed} delay={whole["mean_delay_s"]:.2f}s '
                  f'queue={whole["mean_queue_veh"]:.2f} unfinished={whole["unfinished_vehicles"]} '
                  f'seconds={time.perf_counter()-started:.1f}', flush=True)
    aggregate(root, rows)


def main():
    parser = argparse.ArgumentParser(description='SUMO roadside-radar signal control coursework')
    parser.add_argument('command', choices=('prepare', 'train', 'evaluate', 'plot', 'report', 'all'))
    parser.add_argument('--output', default='results/full')
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--stage-seconds', type=int, default=900)
    parser.add_argument('--test-seeds', type=int, nargs='+', default=[101, 102, 103])
    parser.add_argument('--backend', choices=('auto', 'traci', 'libsumo'), default='auto')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--observation', choices=('full', 'near'), default='full')
    parser.add_argument('--model-seed', type=int, default=7)
    parser.add_argument('--position-noise', type=float, default=0.0)
    parser.add_argument('--speed-noise', type=float, default=0.0)
    args = parser.parse_args()
    if args.episodes < 1 or args.stage_seconds < 1:
        parser.error('episodes and stage-seconds must be positive')
    root = (ROOT / args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = Config(stage_seconds=args.stage_seconds, position_noise_m=args.position_noise,
                    speed_noise_mps=args.speed_noise)
    if args.command in ('train', 'evaluate', 'plot', 'report') and (root / 'config.json').exists():
        stored = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        if stored['stage_seconds'] != config.stage_seconds:
            parser.error('This output directory uses a different stage length; use a new --output.')
    if args.command in ('prepare', 'all'):
        prepare(root, config, args.test_seeds)
    if args.command == 'train':
        if not (root / 'network' / 'network.json').exists():
            prepare(root, config, args.test_seeds)
        train(root, config, args.episodes, args.observation, args.backend, args.model_seed)
    if args.command == 'all':
        train(root, config, args.episodes, 'full', args.backend, args.model_seed)
        if args.ablation:
            train(root, config, args.episodes, 'near', args.backend, args.model_seed)
    if args.command in ('all', 'evaluate'):
        evaluate(root, config, args.test_seeds, args.backend, args.ablation)
    if args.command in ('all', 'plot'):
        from .plots import make_plots
        make_plots(root, config)
    if args.command in ('all', 'report'):
        from .report import write_report
        write_report(root, config)


if __name__ == '__main__':
    main()
