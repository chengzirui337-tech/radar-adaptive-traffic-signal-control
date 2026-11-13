"""Generate an evidence-linked Chinese report from measured experiment outputs."""
from html import escape
import json
from pathlib import Path

import pandas as pd

from .config import PHASES


def write_report(root, config):
    root = Path(root)
    frame = pd.read_csv(root / 'metrics_by_seed.csv')
    all_rows = frame[frame.stage.eq('all')]
    methods = [m for m in ('fixed', 'actuated', 'dqn_full', 'dqn_near') if m in set(frame.method)]
    labels = {'fixed': '定时控制', 'actuated': '感应控制', 'dqn_full': '150 m 轨迹 Double DQN', 'dqn_near': '30 m 近端信息 Double DQN'}
    n = frame.seed.nunique()
    histories = {}
    for method in ('dqn_full', 'dqn_near'):
        p = root / 'training' / f'{method}_history.csv'
        if p.exists():
            histories[method] = pd.read_csv(p)
    full = all_rows[all_rows.method.eq('dqn_full')]
    comparisons = []
    for baseline in ('fixed', 'actuated'):
        b = all_rows[all_rows.method.eq(baseline)]
        change = (1 - full.mean_delay_s.mean() / b.mean_delay_s.mean()) * 100
        direction = '降低' if change >= 0 else '增加'
        comparisons.append(f'相对{labels[baseline]}，150 m DQN 的平均延误{direction} {abs(change):.2f}%。')
    ablation_text = ''
    if 'dqn_near' in methods:
        near = all_rows[all_rows.method.eq('dqn_near')]
        delta = (1 - full.mean_delay_s.mean() / near.mean_delay_s.mean()) * 100
        ablation_text = (f'完整轨迹模型相对 30 m 模型的平均延误变化为 {-delta:+.2f}%。'
                         '该消融比较采用相同模型结构、训练需求种子和训练轮数，近端模型的远端分区及远端统计量全部屏蔽。'
                         '这属于单次模型初始化下的描述性结果，不能据此断言完整轨迹在所有流量和训练条件下均更优。')
    infos = [json.loads(p.read_text(encoding='utf-8')) for p in (root / 'evaluation').glob('*/seed_*/run_info.json')]
    collisions = sum(x['colliding_vehicle_events'] for x in infos)
    teleports = sum(x['teleport_events'] for x in infos)
    unfinished = int(all_rows.unfinished_vehicles.sum())
    training_episodes = len(histories.get('dqn_full', []))
    transition_text = []
    transition_file = root / 'transition_metrics.csv'
    if transition_file.exists():
        transition = pd.read_csv(transition_file)
        for label, name in [('peak_onset', '高峰开始后 180 s'), ('peak_recovery', '高峰结束后 180 s')]:
            means = transition[transition.transition.eq(label)].groupby('method').mean_queue_veh.mean()
            transition_text.append(name + '的跨种子平均排队为：' + '；'.join(
                f'{labels[m]} {means[m]:.2f} veh' for m in methods) + '。这是预先定义的时间窗指标，不能单独识别控制器反应时滞。')
    if 'dqn_full' in histories:
        h = histories['dqn_full']
        transition_text.append(f'150 m DQN 前五轮平均累计奖励为 {h.return_sum.head(5).mean():.2f}，'
                               f'最后五轮为 {h.return_sum.tail(5).mean():.2f}。')
    stage_names = {'low_1': '低流量一', 'normal_1': '平峰一', 'peak': '高峰', 'normal_2': '平峰二', 'low_2': '低流量二'}
    paragraphs = [
        ('作业目标与数据来源', [
            '本作业建立四进口信号交叉口，通过 SUMO 车辆真值模拟路侧毫米波雷达在停车线上游 150 m 内的轨迹输出，并比较定时、感应和强化学习控制。所有车流、轨迹及评价结果均由本地程序实际生成；没有使用真实雷达采集记录，也没有执行雷达点云检测或目标跟踪。',
            f'每个交通阶段持续 {config.stage_seconds / 60:g} min，五阶段共 {config.horizon / 60:g} min。参考需求为交叉口所有进口合计 600、1000、1500、1000、600 veh/h，东西各占 30%，南北各占 20%。这避免了将总需求误用为单进口需求而放大四倍的错误。',
            '按每 60 s 更新到达强度，在基础流量上下 15% 内均匀波动。每个区间的车辆数服从泊松分布，条件于该数量的到达时间服从区间内均匀分布。左直右基础比例依次采用 15/70/15、20/65/15、25/60/15、20/65/15、15/70/15，各比例加入不超过 0.02 的扰动后归一化。']),
        ('交叉口与轨迹状态', [
            '路口每个进口设右转、直行、左转三条专用车道；进口和出口边长约 320 m，道路速度上限为 13.89 m/s。四个保护相位按南北直右、南北左转、东西直右、东西左转循环，不设置行人、非机动车或红灯右转。',
            f'每 1 s 使用 TraCI 兼容接口订阅车辆编号、车道、距停车线距离、速度、加速度、连续等待时间和累计等待时间。默认采用同一 API 的 libsumo 进程内后端以加快训练，也支持纯 TraCI。检测范围以车辆前保险杠到进口车道末端的距离判定，每 {config.bin_m:g} m 划分一格，共 {config.n_bins} 格。',
            f'完整状态共 {config.state_dim} 维：12 车道 × 10 区段 × 车辆数及平均速度，12 车道 × 车辆数、排队数、累计等待时间及平均速度，4 维相位、3 维灯色模式和 1 维持续时间。无车区段速度设为零，各输入按固定尺度归一化。排队定义为速度不高于 0.1 m/s。累计等待时间记忆窗口显式扩展至整个仿真时长，避免使用 SUMO 默认 100 s 窗口而误认为终身累计。']),
        ('强化学习与基准控制', [
            '采用 Double DQN，两个隐藏层分别为 128 和 64 个 ReLU 单元，输出保持当前相位和切换至下一相位两个动作。每 5 s 决策一次。安全状态机规定最短绿灯 10 s、最长绿灯 60 s、黄灯 3 s、全红 1 s；不满足最短绿灯时的切换请求不执行，最长绿灯到达后强制切换。',
            '每秒奖励定义为 −(0.01 W + Q + 2 S)/10，其中 W 是检测区内车辆累计等待时间之和，Q 是检测区内排队车辆数，S 是该秒在检测区内新发生的停车次数。一个决策区间的奖励取各秒奖励的均值；训练累计奖励是全轮所有决策奖励之和。奖励中的 S 为增量，避免将过去停车次数在每一步重复惩罚。',
            '定时控制四相位绿灯分别为 22、10、34、14 s，加入四组黄灯与全红后周期为 96 s，东西与南北总有效绿灯按 60/40 分配。感应控制在最短绿灯后检查当前放行流向 40 m 内是否有车，有车则延长，无车则结束，受相同最大绿灯限制。基准控制每秒判断以保持固定周期精确；DQN 每 5 s 判断。',
            f'DQN 使用 Adam 优化器，学习率 0.0003，折扣因子 0.99，回放容量 50000，批量 64，前 1000 步预热，每 2 个决策更新一次，每 250 次梯度更新同步目标网络。每种模型训练 {training_episodes} 轮，每轮使用不同需求种子，探索率从 1 逐渐降至 0.05，模型初始化种子为 7。使用最后一轮模型，未按测试表现选择检查点。']),
        ('实验设计与指标定义', [
            f'测试需求种子为 {", ".join(map(str, sorted(frame.seed.unique())))}，共 {n} 组。每组的路线文件及车辆随机种子在各控制方法间完全一致；测试种子与训练种子分离。跨方法进行配对比较，报告跨需求种子的均值和样本标准差。',
            '车辆平均延误采用 SUMO tripinfo 的 timeLoss 加 departDelay，平均旅行时间采用 duration 加 departDelay，平均停车次数采用 waitingCount；车辆阶段按计划出发时间归属，避免按完成时间分组造成跨阶段选择偏差。排队均值和最大值以每秒全进口静止车辆总数统计，另报最远排队车辆距停车线的距离。通行量按车辆离开进口停车线的次数统计，非简单以到达出口的次数代替。',
            f'主窗口结束后最多追加 {config.drain_seconds} s 无新增需求的清空期，仅用于完成车辆的全程延误和旅行时间。排队与通行量仍只统计主窗口，清空期不混入。未完成车辆数为 {unfinished}；若该值不为零，完成车辆均值可能有删失偏差，需要延长清空时间。共记录碰撞车辆事件 {collisions} 次、传送事件 {teleports} 次。']),
        ('总体结果', comparisons + ([ablation_text] if ablation_text else [])),
        ('需求变化与训练表现', [
            '交通需求曲线同时给出阶段基础值、波动后的泊松强度和实际每分钟到达数。排队曲线采用 60 s 移动平均展示趋势，原始每秒值保留在 timeseries.csv。阶段延误图以计划出发车辆队列分组，误差线为需求种子间样本标准差。',
            'transition_metrics.csv 分别记录高峰开始后 180 s 和高峰结束后 180 s 的平均及最大排队。高峰开始对应第 30 min，高峰消散对应第 45 min。应同时查看这两个时间窗和整段高峰，不宜仅凭一个全局均值判断响应速度。',
            '训练奖励受需求实现、探索率和策略变化共同影响；奖励曲线改善仅表明该训练过程的回报变化，不能直接等同于已收敛或泛化性能。最终控制效果由独立测试确定。'] + transition_text),
        ('结果边界与可扩展实验', [
            '当前路口几何、驾驶行为和雷达覆盖均为理想化设定，未包含漏检、误检、遮挡、多目标关联或车辆 ID 切换。位置和速度噪声接口已实现，默认实验噪声为零；默认不将未实际运行的噪声实验作为结果。',
            f'{n} 组需求种子可用于课程作业的初步重复实验，但本次每种 DQN 仅训练一个初始化种子，不能量化训练本身的方差，也不作统计显著性或工程部署结论。进一步研究应增加模型初始化种子、测试车流种子、入口不均衡程度和覆盖范围设置，并校准真实路口的饱和流率。',
            '本次 DQN 未超过感应基准。感应控制每秒判断，而 DQN 每 5 s 判断，因此差异同时包含策略和决策频率的影响，不能全部归因于强化学习算法本身。相同安全约束并不意味着两者拥有完全相同的动作时机。',
            '完整轨迹模型仅在全局平均延误上略优于近端模型，两个过渡时窗的平均排队反而略高，说明改善并非在所有阶段都成立。可能原因包括交通总需求偏低、最短绿灯限制、模型样本量、训练随机性及空间信息尚未被有效利用；现有实验不能区分这些机制。']),
    ]
    sections = []
    for title, bodies in paragraphs:
        sections.append(f'<h2>{escape(title)}</h2>' + ''.join(f'<p>{escape(p)}</p>' for p in bodies))
    columns = [('mean_delay_s', '平均延误 s/veh'), ('mean_queue_veh', '平均排队 veh'),
               ('max_queue_veh', '最大排队 veh'), ('mean_stops', '停车次数/veh'),
               ('mean_travel_s', '旅行时间 s/veh'), ('throughput_vehicles', '窗口通行量 veh')]
    def result_table(stage):
        table = '<table><thead><tr><th>控制方法</th>' + ''.join(f'<th>{label}</th>' for _, label in columns) + '</tr></thead><tbody>'
        for m in methods:
            rows = frame[frame.method.eq(m) & frame.stage.eq(stage)]
            table += f'<tr><td>{escape(labels[m])}</td>'
            for col, _ in columns:
                table += f'<td>{rows[col].mean():.2f} ± {rows[col].std():.2f}</td>'
            table += '</tr>'
        return table + '</tbody></table>'
    sections.insert(5, result_table('all'))
    figures = [('01_demand', '图 1 五阶段交通需求'), ('02_queue', '图 2 排队随时间变化'),
               ('03_stage_delay', '图 3 分阶段平均延误'), ('04_training_reward', '图 4 强化学习累计奖励'),
               ('05_signal_phases', '图 5 典型高峰时段信号相位'), ('06_overall_ablation', '图 6 总体控制和信息范围对比')]
    for name, caption in figures:
        sections.append(f'<figure><img src="figures/{name}.png" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    sections.append('<h2>各阶段结果</h2>')
    for stage, name in stage_names.items():
        sections.append(f'<h3>{name}</h3>' + result_table(stage))
    references = [
        ('SUMO 官方安装与下载', 'https://sumo.dlr.de/docs/Downloads.php'),
        ('TraCI 车辆变量和等待时间定义', 'https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html'),
        ('信号控制与相位过渡', 'https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html'),
        ('车辆行程指标 TripInfo', 'https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html'),
        ('Double DQN 原始论文', 'https://arxiv.org/abs/1509.06461'),
    ]
    sections.append('<h2>方法依据与数据文件</h2><p>作业任务来源于用户提供的要求文档。方法和指标定义参见以下一手资料；本文未将作业文档的参考文献列表视作已逐篇核验的证据。</p><ul>' +
                    ''.join(f'<li><a href="{url}">{escape(title)}</a></li>' for title, url in references) + '</ul>')
    sections.append('<p>原始数据：data/seed_*/demand.csv 与 vehicles.csv；雷达轨迹：evaluation/*/seed_101/radar_trajectories.csv.gz；每秒状态统计：evaluation/*/seed_*/timeseries.csv；逐车评价：vehicle_metrics.csv；汇总：summary.csv；配对比较：paired_comparisons.csv；模型：models/*.pt。</p>')
    style = '''body{max-width:1140px;margin:40px auto;padding:0 28px;color:#20252b;font:16px/1.8 "Microsoft YaHei",sans-serif}h1{font-size:28px;line-height:1.4}h2{font-size:21px;margin-top:34px}h3{font-size:18px}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{border:1px solid #d9d9d9;padding:9px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f2f4f5}figure{margin:32px 0}img{max-width:100%}figcaption{font-size:14px;color:#56616a}a{color:#1e637a}@media print{body{margin:0;max-width:none}figure,table{break-inside:avoid}}'''
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>毫米波雷达轨迹与自适应信号控制实验报告</title><style>' + style + '</style><body><h1>毫米波雷达轨迹与自适应信号控制实验报告</h1><p>Python 与 SUMO 可复现实验　仿真数据</p>' + ''.join(sections) + '</body></html>'
    (root / '实验报告.html').write_text(html, encoding='utf-8')
    md = '# 毫米波雷达轨迹与自适应信号控制实验报告\n\n'
    for title, bodies in paragraphs:
        md += f'## {title}\n\n' + '\n\n'.join(bodies) + '\n\n'
        if title == '总体结果':
            for m in methods:
                r = all_rows[all_rows.method.eq(m)]
                md += f'- {labels[m]}：平均延误 {r.mean_delay_s.mean():.2f} ± {r.mean_delay_s.std():.2f} s/veh；平均排队 {r.mean_queue_veh.mean():.2f} veh；平均停车 {r.mean_stops.mean():.2f} 次/veh。\n'
            md += '\n'
    for name, caption in figures:
        md += f'![{caption}](figures/{name}.png)\n\n'
    md += '## 参考资料\n\n' + '\n'.join(f'- [{title}]({url})' for title, url in references)
    (root / '实验报告.md').write_text(md + '\n', encoding='utf-8')
    print(f'Report saved: {root / "实验报告.html"}', flush=True)
