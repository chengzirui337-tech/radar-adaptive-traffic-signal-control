"""Calibrate one global decision bias using validation traffic only."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .config import Config
from .optimization import VALIDATION_SEEDS
from .ppo import PPO
from .predictive_training import evaluate_predictive


def calibrate(root):
    if not (root / 'training_complete.json').exists():
        raise RuntimeError('Finish PPO training before calibration')
    if (root.parent / 'comparison/completed.json').exists():
        raise RuntimeError('Final tests have already run; no further calibration on this experiment')
    raw = root / 'models/best_uncalibrated.pt'
    if not raw.exists():
        shutil.copy2(root / 'models/best.pt', raw)
    agent = PPO()
    metadata = agent.load(raw)
    config = Config(**metadata['config'])
    bias_grid = [-1., -.5, 0., .5, 1.]
    plan = dict(parameter='add bias to switch logit for greedy evaluation', candidates=bias_grid,
                validation_seeds=list(VALIDATION_SEEDS), final_test_used=False,
                selection='minimum validation mean delay, tie favors zero bias')
    (root / 'calibration_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    rows = []
    best_mean, best_bias = metadata['validation_mean_delay_s'], 0.
    existing = pd.read_csv(root / 'validation_history.csv')
    existing = existing[existing.episode.eq(metadata['episodes'])]
    rows.extend(dict(bias=0., **r) for r in existing.to_dict('records'))
    for bias in bias_grid:
        if bias == 0:
            continue
        agent.switch_bias = bias
        reports = evaluate_predictive(root, config, agent, VALIDATION_SEEDS, f'bias_{bias:+.1f}')
        mean = float(np.mean([r['mean_delay_s'] for r in reports]))
        rows.extend(dict(bias=bias, **r) for r in reports)
        pd.DataFrame(rows).to_csv(root / 'calibration_validation.csv', index=False)
        if mean < best_mean:
            best_mean, best_bias = mean, bias
        print(f'CALIBRATION switch_bias={bias:+.1f} validation_delay={mean:.3f}s', flush=True)
    agent.switch_bias = best_bias
    metadata.update(switch_bias=best_bias, validation_mean_delay_s=best_mean,
                    uncalibrated_validation_mean_delay_s=metadata['validation_mean_delay_s'],
                    calibration='five-value global logit bias grid; validation only')
    agent.save(root / 'models/best.pt', metadata)
    (root / 'calibration_selected.json').write_text(json.dumps(dict(bias=best_bias,
        validation_mean_delay_s=best_mean, validation_only=True), indent=2), encoding='utf-8')
    print(f'CALIBRATION selected bias={best_bias:+.1f} validation_delay={best_mean:.3f}s', flush=True)
