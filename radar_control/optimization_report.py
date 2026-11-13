"""Measured optimization results, limitations, plots, and reproducibility links."""
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {'fixed': 'Fixed-time', 'actuated': 'Actuated', 'dqn_40': 'DQN 40 episodes',
          'dqn_longer': 'Longer DQN', 'imitation_only': 'Expert initialization',
          'ppo_uncalibrated': 'PPO before calibration', 'ppo_predictive': 'Cyclic predictive PPO',
          'phase_imitation': 'Phase PPO initialization', 'ppo_phase': 'Phase-selective PPO'}
CHINESE = {'fixed': '定时控制', 'actuated': '感应控制', 'dqn_40': '原 40 轮 DQN',
           'dqn_longer': '长训练 DQN', 'imitation_only': '仅模仿初始化',
           'ppo_uncalibrated': 'PPO 未标定', 'ppo_predictive': '循环切相 PPO',
           'phase_imitation': '直接选相位的模仿初始化', 'ppo_phase': '直接选相位 PPO'}
COLORS = {'fixed': '#8b9298', 'actuated': '#b97933', 'dqn_40': '#718da4',
          'dqn_longer': '#65799a', 'imitation_only': '#ad9b80',
          'ppo_uncalibrated': '#71a09d', 'ppo_predictive': '#658b83',
          'phase_imitation': '#a4a3a0', 'ppo_phase': '#17677d'}


def write_optimization_report(root):
    root = Path(root)
    optimized = root.parent
    frame = pd.read_csv(root / 'metrics_by_seed.csv')
    paired = pd.read_csv(root / 'paired_comparisons.csv').set_index('method')
    freeze = json.loads((root / 'frozen_selection.json').read_text(encoding='utf-8'))
    chosen = freeze['selected_method']
    whole = frame[frame.stage.eq('all')]
    means = whole.groupby('method').mean_delay_s.mean()
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 200})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / f'{name}.png')
        fig.savefig(figures / f'{name}.pdf')
        plt.close(fig)
    seed = int(whole.seed.min())
    demand = pd.read_csv(root / 'data' / f'seed_{seed}' / 'demand.csv')
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for column, label, color in [('base_veh_h', 'Stage baseline', '#8b9298'),
                                  ('demand_veh_h', 'Fluctuating Poisson intensity', '#17677d')]:
        ax.step(np.r_[demand.begin_s.values, 4500] / 60,
                np.r_[demand[column].values, demand[column].iloc[-1]],
                where='post', color=color, label=label)
    ax.set(xlabel='Simulation time (min)', ylabel='Total demand (veh/h)',
           title=f'Five-stage final-test demand | seed {seed}')
    ax.legend(fontsize=8)
    save(fig, '00_final_demand')
    a = pd.read_csv(optimized / 'longer/validation_history.csv').groupby('episode').mean_delay_s.mean()
    b = pd.read_csv(optimized / 'ppo/validation_history.csv').groupby('episode').mean_delay_s.mean()
    c = pd.read_csv(optimized / 'ppo4/validation_history.csv').groupby('episode').mean_delay_s.mean()
    base_val = pd.read_csv(optimized / 'longer/validation_actuated.csv').mean_delay_s.mean()
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.plot(a.index, a.values, '-o', label='DQN warm-start from episode 40', color=COLORS['dqn_longer'], markersize=4)
    ax.plot(b.index, b.values, '-o', label='PPO after expert initialization', color=COLORS['ppo_predictive'], markersize=4)
    ax.plot(c.index, c.values, '-o', label='Phase-selective PPO', color=COLORS['ppo_phase'], markersize=4)
    ax.axhline(base_val, color=COLORS['actuated'], linestyle='--', label='Unchanged actuated baseline')
    ax.set(xlabel='Training episode (DQN includes original 40)', ylabel='Mean validation delay (s/vehicle)',
           title='Model selection uses validation traffic only | seeds 1001–1003')
    ax.legend(fontsize=8)
    save(fig, '01_validation')
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    for i, method in enumerate(LABELS):
        rows = whole[whole.method.eq(method)]
        ax.scatter(np.full(len(rows), i), rows.mean_delay_s, color=COLORS[method], s=13, alpha=.55)
        ax.errorbar(i, rows.mean_delay_s.mean(), yerr=rows.mean_delay_s.std(), fmt='o', color=COLORS[method],
                    markersize=7, capsize=4)
    ax.set(xticks=range(len(LABELS)), xticklabels=['Fixed', 'Actuated', 'DQN 40', 'DQN longer', 'BC cyclic', 'PPO raw', 'PPO tuned', 'BC phase', 'PPO phase'],
           xlabel='Control method', ylabel='Mean delay (s/vehicle)',
           title='Fresh final tests | 10 paired demand seeds; mean ± sample SD')
    ax.tick_params(axis='x', labelsize=8)
    save(fig, '02_final_delays')
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ref = whole[whole.method.eq('actuated')].set_index('seed').mean_delay_s
    for method in ('dqn_40', 'dqn_longer', chosen):
        trial = whole[whole.method.eq(method)].set_index('seed').mean_delay_s.loc[ref.index]
        ax.plot(ref.index, trial - ref, '-o', color=COLORS[method], label=LABELS[method], markersize=4)
    ax.axhline(0, color='#8b9298', linewidth=1)
    ax.set(xticks=ref.index, xlabel='Held-out demand seed', ylabel='Delay minus actuated delay (s/vehicle)',
           title='Paired differences | values below zero favor the learned controller')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, .5))
    save(fig, '03_paired_differences')
    seed = int(whole.seed.min())
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for method in ('fixed', 'actuated', 'dqn_40', chosen):
        ts = pd.read_csv(root / 'evaluation' / method / f'seed_{seed}' / 'timeseries.csv')
        ax.plot(ts.time_s / 60, ts.queue_veh.rolling(60, min_periods=1).mean(), color=COLORS[method],
                label=LABELS[method], linewidth=1.2)
    for t in (15, 30, 45, 60):
        ax.axvline(t, color='#cccccc', linestyle=':', linewidth=.8)
    ax.set(xlabel='Simulation time (min)', ylabel='Queue (vehicles; 60 s moving mean)',
           title=f'Queue under changing demand | new test seed {seed}')
    ax.legend(ncol=2, fontsize=8)
    save(fig, '04_final_queue')
    stages = ['low_1', 'normal_1', 'peak', 'normal_2', 'low_2']
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    display = ['actuated', 'dqn_longer', 'ppo_predictive', 'ppo_phase']
    for i, method in enumerate(display):
        grouped = frame[frame.method.eq(method) & frame.stage.isin(stages)].groupby('stage').mean_delay_s
        ax.bar(np.arange(5) + (i - 1.5) * .2, grouped.mean().reindex(stages), .2,
               yerr=grouped.std().reindex(stages), capsize=2, color=COLORS[method], label=LABELS[method])
    ax.set(xticks=range(5), xticklabels=['Low 1', 'Normal 1', 'Peak', 'Normal 2', 'Low 2'],
           xlabel='Scheduled-departure demand stage', ylabel='Mean delay (s/vehicle)',
           title='Delay by demand stage | mean ± sample SD over 10 test seeds')
    shown = frame[frame.stage.isin(stages) & frame.method.isin(display)].groupby(['method', 'stage']).mean_delay_s
    ax.set_ylim(0, (shown.mean() + shown.std()).max() * 1.25)
    ax.legend(fontsize=8, ncol=2)
    save(fig, '05_final_stage_delay')
    for variant, label, color in [('longer', 'DQN with unchanged reward', COLORS['dqn_longer']),
                                   ('ppo', 'Cyclic PPO with instantaneous delay reward', COLORS['ppo_predictive']),
                                   ('ppo4', 'Phase-selective PPO with instantaneous delay reward', COLORS['ppo_phase'])]:
        h = pd.read_csv(optimized / variant / 'training_history.csv')
        fig, ax = plt.subplots(figsize=(8.4, 3.8))
        ax.plot(h.episode, h.return_sum, color=color, alpha=.3, linewidth=.8)
        ax.plot(h.episode, h.return_sum.rolling(10, min_periods=1).mean(), color=color,
                label='10-episode moving mean')
        ax.set(xlabel='Training episode', ylabel='Episode return (method-specific reward)', title=label)
        ax.legend(fontsize=8)
        save(fig, '06_reward_' + variant)
    phase = pd.read_csv(root / 'evaluation' / chosen / f'seed_{seed}' / 'timeseries.csv')
    phase = phase[phase.time_s.between(1800, 1980)]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    green = phase.signal_mode.eq('green')
    ax.step(phase.time_s / 60, np.where(green, phase.phase, np.nan), where='post',
            linewidth=1.8, color=COLORS[chosen])
    for mode, color, marker in [('yellow', '#b97933', 'o'), ('all_red', '#a34d4d', 'x')]:
        part = phase[phase.signal_mode.eq(mode)]
        ax.scatter(part.time_s / 60, part.phase, s=10, color=color, marker=marker, label=mode.replace('_', ' '))
    ax.set(xlabel='Simulation time (min)', ylabel='Protected phase', yticks=range(4),
           yticklabels=['NS through/right', 'NS left', 'EW through/right', 'EW left'],
           title=f'Selected-controller phases at peak onset | test seed {seed}')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, .5))
    save(fig, '07_final_signal')
    _write_text(root, frame, paired, freeze, means, a, b, base_val)


def _write_text(root, frame, paired, freeze, means, a, b, base_val):
    selected = freeze['selected_method']
    result = paired.loc[selected]
    better = result.mean_difference_s < 0
    verb = '降低' if better else '增加'
    original = freeze['checkpoints']['dqn_40']['metadata']
    long_meta = freeze['checkpoints']['dqn_longer']['metadata']
    ppo_meta = freeze['checkpoints']['ppo_predictive']['metadata']
    phase_meta = freeze['checkpoints']['ppo_phase']['metadata']
    section_data = [
        ('训练轮数与最终结论', [
            f'延长训练有帮助，但不保证持续改善。原 40 轮模型在验证集上的平均延误为 {a.iloc[0]:.3f} s/veh，'
            f'延长训练后的最佳验证均值为 {a.min():.3f} s/veh，感应控制为 {base_val:.3f} s/veh。'
            f'累计训练到 200 轮后，按照验证结果选择第 {long_meta["episodes"]} 轮模型，而非默认使用最后一轮。',
            f'预先依据验证集选出的最终方案是{CHINESE[selected]}。在 10 组新的测试车流上，其平均延误为 '
            f'{means[selected]:.3f} s/veh，感应控制为 {means["actuated"]:.3f} s/veh，平均延误{verb} '
            f'{abs(result.reduction_pct):.2f}%。在 {int(result.better_seed_count)}/10 组种子上延误更低。',
            '结果只适用于当前 SUMO 路口、交通分布和训练初始化。不得将验证集改善替换为最终测试结果，也不代表真实路口已验证。']),
        ('训练量与控制方法的改进', [
            '长训练 DQN 从原 40 轮权重继续训练 160 轮，保持原 5 s 决策间隔、296 维输入、信号约束和奖励定义。由于原始优化器和回放池未保存，本次重新建立 Adam 和回放池，探索率由 0.05 降至 0.01；因此是权重续训实验，不是逐位完全连续的原训练恢复。',
            '轨迹预测 PPO 在原 150 m 分区状态上增加各相位车辆排队、车辆数、40 m 内车辆数、等待量、最近预计到达时间，以及预计 3、6、10 s 内到达的车辆数，合计 344 维。预计到达时间仅用当前检测距离与速度计算，未读取未来车辆或测试数据。',
            'PPO 每秒决定保持或切换，黄灯、全红和最短绿灯期间自动推进，不生成无效决策；最长绿灯仍为 60 s，最短绿灯仍为 10 s，黄灯 3 s、全红 1 s。一次切换到下一个可决策时刻通常持续 14 s，因此折扣和优势估计按实际经过时间计算。',
            'PPO 的奖励改为检测区内每秒速度损失之和，加 0.001 倍累计等待时间的负值，再按实际持续秒数积分并除以 50。该指标更直接惩罚当前延误。它与旧 DQN 奖励定义不同，训练奖励图分别展示，不能横向比较奖励数值大小。',
            f'PPO 使用 4 组训练车流采集感应规则示范，先训练模仿策略与价值估计，再执行 100 轮强化学习。'
            f'验证集选中的 PPO 为第 {ppo_meta["episodes"]} 轮。初始模仿策略单独参与最终测试，便于区分模仿与后续强化学习的贡献。'
            '这属于带专家初始化的强化学习方法，不能称为完全从零训练，也不能把其收益全部归因于增加训练轮数。']),
        ('训练验证测试的隔离', [
            '原 DQN 已训练需求种子为 1–40，续训使用 10001–10160。专家示范使用 30001–30004，PPO 训练使用 50001–50100。所有改进共用验证种子 1001、1002、1003；最终测试种子为 2001–2010。',
            '模型、决策间隔和算法选择都在最终测试前冻结。frozen_selection.json 记录所选方法、每个模型的 SHA256、训练元数据和独立测试种子。最终测试后不再根据结果重新挑选模型。',
            '感应控制规则保持原有 40 m 检测规则及每秒判断，未弱化基准或提高交通需求来制造相对优势。所有方法在每个测试种子下使用相同车辆输入、道路和 SUMO 随机种子。',
            '曾尝试带预计到达特征的半马尔可夫 DQN，但前 7 轮训练回报退化，已停止并保留日志；该试验不作为成功结果，也未接触最终测试集。']),
        ('评价与不确定性', [
            '每组正式仿真仍为 5 段各 15 min、合计 75 min；需求合计为 600、1000、1500、1000、600 veh/h，东西占 60%，每分钟需求和转向比例随机波动。主窗口后追加清空期，车辆延误统计包含入口排队，排队和通行量仍仅统计主窗口。',
            '总体和阶段指标来自逐车 tripinfo 与每秒 timeseries，延误为 timeLoss 加 departDelay，旅行时间为 duration 加 departDelay，停车次数为 waitingCount。阶段按车辆计划出发时间划分。',
            f'最终所选方案与感应控制的种子配对延误差为 {result.mean_difference_s:+.3f} s/veh，'
            f'配对需求种子自助抽样的 95% 区间为 [{result.bootstrap_ci_low_s:+.3f}, {result.bootstrap_ci_high_s:+.3f}] s/veh。'
            '该区间仅反映这一个已训练模型对需求随机性的敏感程度；只有一个训练初始化，不能替代多个模型独立训练后的区间。',
            '改进方法同时改变了输入特征、奖励、算法、初始化和决策间隔。因此只能将整体结果解释为组合改进的收益。长训练 DQN 实验用于单独判断增加训练量的作用，其余改动仍需进一步消融才能分配贡献。'])
    ]
    bias = ppo_meta.get('switch_bias', 0.)
    section_data[1][1].append(
        f'强化学习训练后，另在验证集上用 −1、−0.5、0、0.5、1 五个全局切换得分偏置做标定，选中偏置为 {bias:+.1f}。'
        '最终同时测试未标定和标定后的 PPO，防止将决策阈值调整的作用归入强化学习训练。'
        f'仅模仿、未标定 PPO、标定后 PPO 的新测试均值分别为 {means["imitation_only"]:.3f}、'
        f'{means["ppo_uncalibrated"]:.3f}、{means["ppo_predictive"]:.3f} s/veh。')
    section_data[1][1].append(
        f'直接选相位版本在相同轨迹输入和 PPO 学习方法上，将动作扩展为 4 个目标放行相位；'
        f'选择当前相位表示保持，选择其他相位时执行同样的黄灯、全红和最短绿灯。该版本训练 60 轮，'
        f'验证集选中第 {phase_meta["episodes"]} 轮，不再要求按固定顺序经过空闲相位。'
        '为防止某方向长期得不到放行，检测区内某个非当前相位的车辆连续等待达到 90 s 时，'
        '安全层优先请求等待最久的相位；该保护仅使用当前可观测的车辆等待数据。'
        f'直接选相位版本的模仿初始化与强化学习后，新测试延误分别为 '
        f'{means["phase_imitation"]:.3f} 和 {means["ppo_phase"]:.3f} s/veh。'
        '因此，动作空间变化和等待保护也属于组合改进的一部分；不能把整体差异全部解释成训练轮数的效果。')
    section_data[2][1].append('直接选相位版本使用训练需求种子 60001–60060，也只使用原先保留的 1001–1003 验证集选模；最终测试仍为 2001–2010。')
    table_columns = [('mean_delay_s', '延误 s/veh'), ('mean_queue_veh', '平均排队 veh'),
                     ('max_queue_veh', '最大排队 veh'), ('mean_stops', '停车次数/veh'),
                     ('mean_travel_s', '旅行时间 s/veh'), ('throughput_vehicles', '窗口通行量 veh')]
    def table(stage='all'):
        content = '<table><thead><tr><th>控制方法</th>' + ''.join(f'<th>{n}</th>' for _, n in table_columns) + '</tr></thead><tbody>'
        for method in LABELS:
            rows = frame[frame.method.eq(method) & frame.stage.eq(stage)]
            content += f'<tr><td>{CHINESE[method]}</td>' + ''.join(
                f'<td>{rows[c].mean():.2f} ± {rows[c].std():.2f}</td>' for c, _ in table_columns) + '</tr>'
        return content + '</tbody></table>'
    html = []
    md = '# 延长训练与信号控制优化报告\n\n'
    for i, (title, paragraphs) in enumerate(section_data):
        html.append(f'<h2>{title}</h2>' + ''.join(f'<p>{escape(p)}</p>' for p in paragraphs))
        md += f'## {title}\n\n' + '\n\n'.join(paragraphs) + '\n\n'
        if i == 0:
            html.append(table())
            for method in LABELS:
                md += f'- {CHINESE[method]}：平均延误 {means[method]:.3f} s/veh。\n'
            md += '\n'
    figure_captions = [('00_final_demand', '最终测试的五阶段需求'), ('01_validation', '验证集延误与训练轮数'), ('02_final_delays', '新测试车流的总体延误'),
                       ('03_paired_differences', '每组车流相对感应控制的延误差'), ('04_final_queue', '典型测试车流排队曲线'),
                       ('05_final_stage_delay', '分需求阶段延误'), ('06_reward_longer', '长训练 DQN 回报'),
                       ('06_reward_ppo', '循环切相 PPO 训练回报'), ('06_reward_ppo4', '直接选相位 PPO 训练回报'),
                       ('07_final_signal', '最终控制器高峰起始时段相位变化')]
    for name, caption in figure_captions:
        html.append(f'<figure><img src="comparison/figures/{name}.png" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
        md += f'![{caption}](comparison/figures/{name}.png)\n\n'
    references = [('Double DQN 原始论文', 'https://arxiv.org/abs/1509.06461'),
                  ('PPO 原始论文', 'https://arxiv.org/abs/1707.06347'),
                  ('SUMO TraCI 官方说明', 'https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html')]
    html.append('<h2>参考与复现</h2><ul>' + ''.join(f'<li><a href="{u}">{n}</a></li>' for n, u in references) + '</ul>')
    html.append('<p>复现命令和各版本差异见项目 README 的优化训练部分。逐车指标、每秒排队、模型、训练与验证历史、模型冻结清单均保留在 results/optimized 下。</p>')
    style = 'body{max-width:1140px;margin:40px auto;padding:0 28px;font:16px/1.8 "Microsoft YaHei",sans-serif;color:#222}h1{font-size:28px}h2{font-size:21px;margin-top:32px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#f3f4f5}figure{margin:30px 0}img{max-width:100%}figcaption{font-size:14px;color:#555}a{color:#246c80}'
    path = root.parent / '优化训练报告.html'
    path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>延长训练与信号控制优化报告</title><style>' + style + '</style><body><h1>延长训练与信号控制优化报告</h1>' + ''.join(html) + '</body></html>', encoding='utf-8')
    md += '## 方法来源\n\n' + '\n'.join(f'- [{n}]({u})' for n, u in references)
    (root.parent / '优化训练报告.md').write_text(md + '\n', encoding='utf-8')
    print(f'Optimization report saved: {path}', flush=True)
