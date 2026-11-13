"""Freeze validation-selected models, then evaluate a fresh ten-seed test set."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .dqn import DQN
from .experiment import prepare
from .metrics import summarize
from .optimization import FINAL_TEST_SEEDS, VALIDATION_SEEDS, run_policy
from .ppo import PPO
from .predictive_training import evaluate_predictive

ROOT = Path(__file__).resolve().parents[1]


def finalize():
    optimized = ROOT / 'results/optimized'
    for variant in ('longer', 'ppo', 'ppo4'):
        if not (optimized / variant / 'training_complete.json').exists():
            raise RuntimeError(f'{variant} has not completed training; final test remains sealed.')
    root = optimized / 'comparison'
    config = Config()
    prepare(root, config, FINAL_TEST_SEEDS)
    checkpoints = {
        'dqn_40': ROOT / 'results/full/models/dqn_full.pt',
        'dqn_longer': optimized / 'longer/models/best.pt',
        'imitation_only': optimized / 'ppo/models/episode_000.pt',
        'ppo_uncalibrated': optimized / 'ppo/models/best_uncalibrated.pt',
        'ppo_predictive': optimized / 'ppo/models/best.pt',
        'phase_imitation': optimized / 'ppo4/models/episode_000.pt',
        'ppo_phase': optimized / 'ppo4/models/best.pt'}
    agents, metadata = {}, {}
    for method, path in checkpoints.items():
        if method in ('phase_imitation', 'ppo_phase'):
            agent = PPO(action_count=4)
        else:
            agent = PPO() if method in ('imitation_only', 'ppo_uncalibrated', 'ppo_predictive') else DQN(config)
        meta = agent.load(path)
        training_seeds = set(meta['demand_seeds']) | set(meta.get('expert_seeds', []))
        assert not training_seeds.intersection(FINAL_TEST_SEEDS)
        assert not set(VALIDATION_SEEDS).intersection(FINAL_TEST_SEEDS)
        agents[method], metadata[method] = agent, meta
    # Write the freeze manifest before touching final evaluation outcomes.
    selected = min(('dqn_longer', 'ppo_predictive', 'ppo_phase'), key=lambda m: metadata[m]['validation_mean_delay_s'])
    freeze = dict(selection_rule='minimum mean validation delay', selected_method=selected,
                  validation_seeds=list(VALIDATION_SEEDS), final_test_seeds=list(FINAL_TEST_SEEDS),
                  selected_before_final_test=True, baseline_unchanged=True,
                  checkpoints={m: dict(path=str(path.relative_to(ROOT)),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(), metadata=metadata[m])
                      for m, path in checkpoints.items()})
    (root / 'frozen_selection.json').write_text(json.dumps(freeze, indent=2), encoding='utf-8')
    rows = []
    for method in ('fixed', 'actuated', 'dqn_40', 'dqn_longer', 'imitation_only', 'ppo_uncalibrated', 'ppo_predictive', 'phase_imitation', 'ppo_phase'):
        if method in ('imitation_only', 'ppo_uncalibrated', 'ppo_predictive', 'phase_imitation', 'ppo_phase'):
            c = replace(config, decision_seconds=1)
            reports = evaluate_predictive(root, c, agents[method], FINAL_TEST_SEEDS, method, record=True)
        else:
            c = config
            reports = []
            for seed in FINAL_TEST_SEEDS:
                reports.append(run_policy(root, c, seed, method, agents.get(method), split='evaluation', record=True))
                print(f'FINAL {method} seed={seed} delay={reports[-1]["mean_delay_s"]:.3f}s', flush=True)
        for report in reports:
            seed = report['seed']
            assert report['collisions'] == report['teleports'] == 0
            measures = summarize(root / 'evaluation' / method / f'seed_{seed}',
                                 root / 'data' / f'seed_{seed}', c, method, seed)
            assert measures[0]['unfinished_vehicles'] == 0
            rows.extend(measures)
        pd.DataFrame(rows).to_csv(root / 'metrics_by_seed.csv', index=False)
        print(f'FINAL {method} mean={np.mean([r["mean_delay_s"] for r in reports]):.3f}s', flush=True)
    frame = pd.DataFrame(rows)
    columns = ['mean_delay_s', 'mean_queue_veh', 'max_queue_veh', 'mean_stops', 'mean_travel_s',
               'throughput_vehicles', 'max_lane_queue_m']
    summary = frame.groupby(['method', 'stage'])[columns].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(root / 'summary.csv', index=False)
    paired = []
    all_rows = frame[frame.stage.eq('all')]
    reference = all_rows[all_rows.method.eq('actuated')].set_index('seed')
    rng = np.random.default_rng(20261112)
    for method in ('dqn_40', 'dqn_longer', 'imitation_only', 'ppo_uncalibrated', 'ppo_predictive', 'phase_imitation', 'ppo_phase'):
        trial = all_rows[all_rows.method.eq(method)].set_index('seed').loc[reference.index]
        differences = trial.mean_delay_s.values - reference.mean_delay_s.values
        boot = rng.choice(differences, size=(20000, len(differences)), replace=True).mean(axis=1)
        paired.append(dict(method=method, baseline='actuated', paired_seeds=len(differences),
                           mean_difference_s=float(differences.mean()), std_difference_s=float(differences.std(ddof=1)),
                           reduction_pct=float((1 - trial.mean_delay_s.mean() / reference.mean_delay_s.mean()) * 100),
                           bootstrap_ci_low_s=float(np.quantile(boot, .025)),
                           bootstrap_ci_high_s=float(np.quantile(boot, .975)),
                           better_seed_count=int((differences < 0).sum())))
    pd.DataFrame(paired).to_csv(root / 'paired_comparisons.csv', index=False)
    (root / 'completed.json').write_text(json.dumps(dict(selected_method=selected, seed_count=10,
        runs=90, no_test_based_reselection=True), indent=2), encoding='utf-8')
    from .optimization_report import write_optimization_report
    write_optimization_report(root)
