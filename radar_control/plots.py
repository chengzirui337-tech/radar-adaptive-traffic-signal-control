"""Reproducible coursework figures from saved outputs, never invented results."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {'fixed': 'Fixed-time', 'actuated': 'Actuated', 'dqn_full': 'DQN 150 m', 'dqn_near': 'DQN 30 m'}
COLORS = {'fixed': '#606c76', 'actuated': '#d48436', 'dqn_full': '#287c8e', 'dqn_near': '#855b99'}


def make_plots(root, config):
    root = Path(root)
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.dpi': 200})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / f'{name}.png')
        fig.savefig(figures / f'{name}.pdf')
        plt.close(fig)

    metrics = pd.read_csv(root / 'metrics_by_seed.csv')
    methods = [m for m in LABELS if m in set(metrics.method)]
    seed = int(metrics.seed.min())
    demand = pd.read_csv(root / 'data' / f'seed_{seed}' / 'demand.csv')
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for column, label, color in [('base_veh_h', 'Stage baseline', '#606c76'),
                                  ('demand_veh_h', 'Fluctuating Poisson rate', '#287c8e')]:
        x = np.r_[demand.begin_s.values, config.horizon] / 60
        y = np.r_[demand[column].values, demand[column].iloc[-1]]
        ax.step(x, y, where='post', label=label, color=color, linewidth=1.4)
    ax.scatter((demand.begin_s + demand.end_s) / 120,
               demand.generated_vehicles * 3600 / (demand.end_s - demand.begin_s),
               s=10, color='#d48436', label='Realized count per interval (veh/h)', alpha=.6)
    ax.set(xlabel='Simulation time (min)', ylabel='All-approach demand (veh/h)',
           title=f'Five-stage traffic demand | held-out seed {seed}')
    ax.legend(fontsize=8, loc='upper right')
    save(fig, '01_demand')

    series = {m: pd.read_csv(root / 'evaluation' / m / f'seed_{seed}' / 'timeseries.csv') for m in methods}
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for m, frame in series.items():
        ax.plot(frame.time_s / 60, frame.queue_veh.rolling(60, min_periods=1).mean(),
                label=LABELS[m], color=COLORS[m], linewidth=1.2)
    for boundary in range(1, 5):
        ax.axvline(boundary * config.stage_seconds / 60, color='#bbbbbb', linestyle=':', linewidth=.8)
    ax.set(xlabel='Simulation time (min)', ylabel='Total queue (vehicles; 60 s moving mean)',
           title=f'Queue response to changing demand | held-out seed {seed}')
    ax.legend(ncol=2, fontsize=8)
    save(fig, '02_queue')

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    stages = ['low_1', 'normal_1', 'peak', 'normal_2', 'low_2']
    width = .8 / len(methods)
    x = np.arange(5)
    for i, m in enumerate(methods):
        group = metrics[(metrics.method == m) & metrics.stage.isin(stages)].groupby('stage').mean_delay_s
        mean = group.mean().reindex(stages)
        std = group.std().reindex(stages).fillna(0)
        ax.bar(x - .4 + width * (i + .5), mean, width, yerr=std, capsize=2,
               color=COLORS[m], label=LABELS[m])
    ax.set(xticks=x, xticklabels=['Low 1', 'Normal 1', 'Peak', 'Normal 2', 'Low 2'],
           xlabel='Demand stage (scheduled-departure cohort)', ylabel='Mean delay (s/vehicle)',
           title=f'Delay by demand stage | mean ± sample SD over {metrics.seed.nunique()} paired demand seeds')
    ax.set_ylim(0, metrics[metrics.stage.isin(stages)].mean_delay_s.max() * 1.25)
    ax.legend(ncol=2, fontsize=8)
    save(fig, '03_stage_delay')

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for variant in ['dqn_full', 'dqn_near']:
        history_path = root / 'training' / f'{variant}_history.csv'
        if not history_path.exists():
            continue
        frame = pd.read_csv(history_path)
        ax.plot(frame.episode, frame.return_sum, alpha=.25, color=COLORS[variant], linewidth=.7)
        ax.plot(frame.episode, frame.return_sum.rolling(5, min_periods=1).mean(),
                label=LABELS[variant] + ' (5-episode moving mean)', color=COLORS[variant])
    ax.set(xlabel='Training episode (new demand seed each episode)', ylabel='Sum of decision rewards',
           title='Training returns under changing exploration and demand')
    ax.legend(fontsize=8)
    save(fig, '04_training_reward')

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    frame = series['dqn_full']
    begin, end = 2 * config.stage_seconds, min(2 * config.stage_seconds + 180, config.horizon)
    frame = frame[frame.time_s.between(begin, end)]
    green = frame.signal_mode.eq('green')
    ax.step(frame.time_s / 60, np.where(green, frame.phase, np.nan), where='post', color=COLORS['dqn_full'], linewidth=1.8)
    yellow = frame.signal_mode.eq('yellow')
    red = frame.signal_mode.eq('all_red')
    ax.scatter(frame.loc[yellow, 'time_s'] / 60, frame.loc[yellow, 'phase'], s=7, color='#d48436', label='Yellow')
    ax.scatter(frame.loc[red, 'time_s'] / 60, frame.loc[red, 'phase'], s=10, marker='x', color='#9d4548', label='All-red')
    ax.set(yticks=range(4), yticklabels=['NS through/right', 'NS left', 'EW through/right', 'EW left'],
           xlabel='Simulation time (min)', ylabel='Protected phase',
           title=f'DQN phase decisions during peak onset | held-out seed {seed}')
    ax.legend(fontsize=8)
    save(fig, '05_signal_phases')

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    whole = metrics[metrics.stage.eq('all')]
    for m in methods:
        rows = whole[whole.method.eq(m)]
        ax.errorbar(LABELS[m], rows.mean_delay_s.mean(), yerr=rows.mean_delay_s.std(),
                    fmt='o', color=COLORS[m], capsize=5, markersize=8)
        ax.scatter([LABELS[m]] * len(rows), rows.mean_delay_s, s=16, color=COLORS[m], alpha=.5)
    ax.set(xlabel='Control method and observation range', ylabel='Mean delay (s/vehicle)',
           title='Whole-horizon comparison | dots are seeds; bars are sample SD')
    save(fig, '06_overall_ablation')

    rows = []
    for m in methods:
        for s in sorted(metrics.seed.unique()):
            ts = pd.read_csv(root / 'evaluation' / m / f'seed_{s}' / 'timeseries.csv')
            for label, boundary in [('peak_onset', 2 * config.stage_seconds), ('peak_recovery', 3 * config.stage_seconds)]:
                part = ts[ts.time_s.between(boundary + 1, min(boundary + 180, config.horizon))]
                rows.append(dict(method=m, seed=s, transition=label, window_s=180,
                                 mean_queue_veh=part.queue_veh.mean(), max_queue_veh=part.queue_veh.max()))
    pd.DataFrame(rows).to_csv(root / 'transition_metrics.csv', index=False)
