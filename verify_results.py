"""Audit the completed experiment against source data and signal constraints."""
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.packages'))
import numpy as np
import pandas as pd


def verify(root=ROOT / 'results' / 'full'):
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(root / 'metrics_by_seed.csv')
    reports = []
    for method in frame.method.unique():
        for seed in sorted(frame.seed.unique()):
            folder = root / 'evaluation' / method / f'seed_{seed}'
            ts = pd.read_csv(folder / 'timeseries.csv')
            info = json.loads((folder / 'run_info.json').read_text(encoding='utf-8'))
            assert len(ts) == config['stage_seconds'] * 5
            assert np.array_equal(ts.time_s, np.arange(1, len(ts) + 1))
            assert info['colliding_vehicle_events'] == info['teleport_events'] == 0
            groups = ts.groupby((ts.signal_mode.ne(ts.signal_mode.shift()) | ts.phase.ne(ts.phase.shift())).cumsum())
            blocks = list(groups)
            for i, (_, block) in enumerate(blocks):
                mode = block.signal_mode.iloc[0]
                if mode == 'yellow' and i < len(blocks) - 1:
                    assert len(block) == config['yellow'], (method, seed, 'yellow', len(block))
                if mode == 'all_red' and i < len(blocks) - 1:
                    assert len(block) == config['all_red'], (method, seed, 'all_red', len(block))
                if mode == 'green' and i < len(blocks) - 1:
                    assert config['min_green'] <= len(block) <= config['max_green'], (method, seed, len(block))
            trips = pd.read_csv(folder / 'vehicle_metrics.csv')
            manifest = pd.read_csv(root / 'data' / f'seed_{seed}' / 'vehicles.csv')
            assert len(trips) == len(manifest)
            assert trips.complete.all(), (method, seed, 'unfinished trips')
            demand = pd.read_csv(root / 'data' / f'seed_{seed}' / 'demand.csv')
            assert demand.generated_vehicles.sum() == len(manifest)
            np.testing.assert_allclose(demand[['left_ratio', 'straight_ratio', 'right_ratio']].sum(axis=1), 1)
            metrics = frame[(frame.method == method) & (frame.seed == seed) & (frame.stage == 'all')].iloc[0]
            np.testing.assert_allclose(metrics.mean_delay_s, trips.delay_s.mean())
            np.testing.assert_allclose(metrics.mean_queue_veh, ts.queue_veh.mean())
            radar = folder / 'radar_trajectories.csv.gz'
            count = 0
            if radar.exists():
                for chunk in pd.read_csv(radar, chunksize=50000):
                    assert chunk.distance_m.between(0, 150).all()
                    assert chunk.vehicle_id.isin(manifest.vehicle_id).all()
                    count += len(chunk)
            npz = folder / 'decision_transitions.npz'
            if method.startswith('dqn'):
                with np.load(npz) as data:
                    assert data['states'].shape == (len(ts) // config['decision_seconds'], 296)
                    assert data['dones'][-1] and not data['dones'][:-1].any()
            reports.append(dict(method=method, seed=int(seed), vehicle_count=len(trips),
                                trajectory_rows=count, seconds=len(ts), safety_passed=True))
    result = dict(passed=True, checks=['main duration', 'signal durations', 'coverage', 'collision/teleport',
                                     'complete trip cohorts', 'demand conservation', 'metric recomputation',
                                     'DQN transition dimensions'], runs=reports)
    (root / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
