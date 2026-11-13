"""Audit held-out comparisons and duration-aware training transitions."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.packages'))
import numpy as np
import pandas as pd


def main():
    root = ROOT / 'results/optimized/comparison'
    frozen = json.loads((root / 'frozen_selection.json').read_text(encoding='utf-8'))
    metrics = pd.read_csv(root / 'metrics_by_seed.csv')
    assert frozen['selected_before_final_test'] and frozen['baseline_unchanged']
    for method, item in frozen['checkpoints'].items():
        path = ROOT / item['path'].replace('\\', '/')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        seeds = set(item['metadata']['demand_seeds']) | set(item['metadata'].get('expert_seeds', []))
        assert not seeds.intersection(frozen['final_test_seeds'])
    verified = []
    for method in metrics.method.unique():
        for seed in frozen['final_test_seeds']:
            folder = root / 'evaluation' / method / f'seed_{seed}'
            ts = pd.read_csv(folder / 'timeseries.csv')
            info = json.loads((folder / 'run_info.json').read_text(encoding='utf-8'))
            assert len(ts) == 4500 and np.array_equal(ts.time_s, np.arange(1, 4501))
            assert info['colliding_vehicle_events'] == info['teleport_events'] == 0
            groups = list(ts.groupby((ts.signal_mode.ne(ts.signal_mode.shift()) | ts.phase.ne(ts.phase.shift())).cumsum()))
            for i, (_, block) in enumerate(groups[:-1]):
                mode, length = block.signal_mode.iloc[0], len(block)
                if mode == 'green':
                    assert 10 <= length <= 60, (method, seed, length)
                elif mode == 'yellow':
                    assert length == 3
                else:
                    assert mode == 'all_red' and length == 1
            vehicles = pd.read_csv(folder / 'vehicle_metrics.csv')
            manifest = pd.read_csv(root / 'data' / f'seed_{seed}' / 'vehicles.csv')
            assert len(vehicles) == len(manifest) and vehicles.complete.all()
            row = metrics[(metrics.method == method) & (metrics.seed == seed) & (metrics.stage == 'all')].iloc[0]
            np.testing.assert_allclose(row.mean_delay_s, vehicles.delay_s.mean())
            np.testing.assert_allclose(row.mean_queue_veh, ts.queue_veh.mean())
            transition = folder / 'decision_transitions.npz'
            if method not in ('fixed', 'actuated'):
                with np.load(transition) as data:
                    predictive = method in ('imitation_only', 'ppo_uncalibrated', 'ppo_predictive', 'phase_imitation', 'ppo_phase')
                    assert data['states'].shape[1] == (344 if predictive else 296)
                    assert np.isfinite(data['states']).all()
                    assert data['dones'][-1] and not data['dones'][:-1].any()
                    np.testing.assert_array_equal(data['next_states'][:-1], data['states'][1:])
                    if predictive:
                        assert data['duration_s'].sum() == 4490
                        np.testing.assert_allclose(data['discounts'], .99 ** (data['duration_s'] / 5))
                        np.testing.assert_allclose(data['rewards'].sum(), ts[ts.time_s > 10].reward.sum() / 5)
            verified.append(dict(method=method, seed=int(seed), vehicles=len(vehicles), safety_passed=True))
    result = dict(passed=True, runs=len(verified), seed_count=10, frozen_checkpoints_verified=True,
                  safety_intervals_verified=True, unfinished_vehicles=0, metric_recomputation_passed=True,
                  semi_markov_transitions_verified=True, details=verified)
    (root / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}, indent=2))


if __name__ == '__main__':
    main()
