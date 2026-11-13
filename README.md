# 毫米波雷达轨迹与交叉口自适应信号控制

基于 Python、SUMO、Double DQN 和 PPO 的单点交叉口信号控制课程项目。包含完整源代码、已训练模型、仿真需求、测试数据、探索过程和可编辑的 Word 报告。

- [Word 课程报告（DQN 与 PPO）](毫米波雷达轨迹感知自适应信号控制课程报告_DQN与PPO.docx)
- [DQN 初始实验报告](results/full/实验报告.html)
- [延长训练与 PPO 优化报告](results/optimized/优化训练报告.html)
- [最终测试汇总](results/optimized/comparison/summary.csv)
- [数据说明](数据说明.md) · [作业要求与数据方案](作业要求与数据方案.md)

HTML 报告需先下载或克隆仓库，再在本地浏览器打开。GitHub 的 Code → Download ZIP 可以下载整个项目。

本仓库数据由 SUMO 生成，用于模拟理想毫米波雷达车辆轨迹，并非真实雷达采集数据。最终 PPO 在 10 组预留需求种子中平均延误较感应控制降低 7.28%；此结论限于当前仿真设置和单个训练初始化。

## 延长训练与后续优化

原 40 轮实验保留在 `results/full`。优化实验单独保存在 `results/optimized`，打开 `results/optimized/优化训练报告.html` 查看正式结果。最终直接选相位 PPO 在 10 组新车流上的平均延误为 21.964 s/veh，感应控制为 23.689 s/veh，降低 7.28%，10 组种子均改善。最终模型位于 `results/optimized/ppo4/models/best.pt`；这是当前仿真设置与单个训练初始化的结果。

```powershell
python optimize.py longer --episodes 160
python optimize.py ppo --episodes 100 --output results/optimized/ppo
python optimize.py calibrate
python optimize.py ppo4 --episodes 60 --output results/optimized/ppo4
python optimize.py finalize
python verify_optimized.py
```

也可以执行 `.\运行作业.ps1 optimize`。它会执行较长训练；仅查看已有报告直接运行 `.\运行作业.ps1`。

长训练分支从 40 轮权重续训 160 轮，维持原 5 s 动作时机和奖励。优化器和回放池重新建立，探索率从 0.05 降至 0.01。每 20 轮用固定验证车流选择模型。

另一分支是带感应规则专家初始化的 PPO：344 维状态包含原 150 m 区段信息与预计到达特征，在允许调整绿灯时每秒决策；黄灯、全红、最短绿灯期间自动推进，并按实际动作时长计算折扣与优势。它改变了算法、奖励、输入、初始化和决策时机，不能将所有收益解释为增加训练轮数。仅模仿初始化的策略也单独测试。

训练后额外用验证集标定一个全局切换得分偏置，候选为 −1、−0.5、0、0.5、1。最终同时保留未标定 PPO 和标定 PPO 的测试结果，避免混淆训练与阈值调整的作用。

另一个 PPO 分支直接从 4 个目标相位中选择放行相位，可以跳过空闲方向，训练 60 轮。选择当前相位时保持绿灯，换相位时仍执行黄灯、全红和最短绿灯。若检测区非当前相位有车连续等待超过 90 s，优先服务等待最久的相位，避免自由选相位时饿死低流量方向。该分支也保留仅模仿初始化的对照，结果不能全部归因于训练轮数。最终所选模型在测试前按验证均值冻结。

验证种子固定为 1001–1003；最终测试种子预留为 2001–2010，测试前冻结检查点并记录 SHA256。感应基准沿用原 40 m 规则及每秒判断，未弱化。最终结果报告种子配对差值和需求随机性的自助抽样区间，但仅训练一个模型初始化，不能当作训练方差的估计。

`results/optimized/predictive` 保留一个训练初期退化并停止的 DQN 试验，不计入最终成功模型。所有正式结果均来自实际仿真。

查看与重绘已交付结果可执行 `python optimize.py report`。完整重训请在新的项目副本中使用空的 `results/optimized` 目录；已完成最终测试的实验禁止继续标定，避免把已见测试结果用于调参。

仓库保留需求数据、训练历史、模型检查点和全部验证及测试结果；每轮训练的中间行程 XML 与日志未上传，可重新运行实验生成。

## 已生成的数据与结果

- `results/full/data/`：交通需求、逐车到达与转向、SUMO 路线、数据来源与随机种子。
- `results/full/network/`：四进口三车道路网与连接、相位映射。
- `results/full/evaluation/`：三种控制及近端消融的逐车评价、每秒统计和第一个测试种子的模拟雷达轨迹。
- `results/full/training/`：训练历史；每轮中间行程文件不随仓库分发。
- `results/full/models/`：训练后的 Double DQN 权重。
- `results/full/figures/`：PNG 和矢量 PDF 图。
- `results/full/summary.csv`：跨需求种子的均值、样本标准差和样本量。
- `results/full/paired_comparisons.csv`：同一需求种子下的延误差值。
- `results/full/实验报告.html`：完整报告，浏览器打开即可查看。

所有数据均为本地生成的 SUMO 仿真数据，不能标注为实测雷达数据。仿真结果以实际运行生成的文件为准。

## 运行环境

建议使用 Python 3.12。SUMO 1.28.0 和接口通过 `requirements.txt` 安装；本仓库不分发第三方软件运行环境。实际实验版本见 `results/full/versions.json`。

```powershell
git clone https://github.com/chengzirui337-tech/radar-adaptive-traffic-signal-control.git
cd radar-adaptive-traffic-signal-control
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

只查看已有结果可直接打开 Word 报告，或执行 `.\运行作业.ps1` 打开 HTML 报告。Linux/macOS 下可用 `source .venv/bin/activate` 激活环境，并直接运行 Python 命令；当前交付结果和复核流程是在 Windows 上生成和验证的。

已有 `results` 保存正式实验结果。重新完整训练时，请使用新的项目副本并将其 `results` 目录移走，避免覆盖已交付结果或复用已见测试集。

## 分步运行

```powershell
python run.py prepare
python run.py train --episodes 40 --observation full
python run.py train --episodes 40 --observation near
python run.py evaluate --ablation --test-seeds 101 102 103
python run.py plot
python run.py report
python -m unittest discover -s tests -v
```

默认每阶段 900 s，共 4500 s。快速调试可加 `--stage-seconds 60 --output results/smoke`，但这种短实验不能当作满足作业时长要求的正式实验。所有分步命令需要使用一致的配置。

默认使用速度较快的 libsumo，其接口与 TraCI 一致。可以用 `--backend traci` 改为独立 SUMO 进程和纯 TraCI 接口。中文路径下使用相对文件名，并避免向 Windows SUMO 传入非 ASCII 的绝对 SUMO_HOME。

每轮训练使用不同需求种子；测试种子不得与训练重叠。如果将训练轮数增大到 100 或以上，请同时把测试种子改为更大且互不重叠的值，例如 1001、1002、1003。

## 数据字段

模拟雷达文件是 gzip 压缩的 CSV，可直接用 Pandas 读取：

```python
import pandas as pd
tracks = pd.read_csv('results/full/evaluation/fixed/seed_101/radar_trajectories.csv.gz')
print(tracks.head())
```

字段依次为：`time_s` 仿真秒、`vehicle_id` 车辆 ID、`approach` 进口、`lane_id` 车道、`distance_m` 距停车线距离、`speed_mps` 速度、`acceleration_mps2` 加速度、`waiting_s` 连续等待、`accumulated_waiting_s` 全程累计等待、`phase` 保护相位、`signal_mode` 灯色模式。相位 0～3 分别为南北直右、南北左转、东西直右、东西左转。

原始需求和结果 CSV 均为 UTF-8，可用 Python 或 Excel 导入。完整模型的 `decision_transitions.npz` 保存训练状态格式的决策数据，用 `numpy.load` 读取 `states`、`actions`、`rewards`、`next_states`、`dones`。

## 方法与局限

三种方法共享同一信号安全状态机。定时和感应基准每秒判断；DQN 每 5 s 判断。报告中明确说明指标口径、主窗口与清空期、未完成车辆、相同需求种子配对比较及训练初始化数量。完整轨迹消融不预设一定获胜。位置和速度噪声参数已经实现，默认结果为无噪声；未经实际运行的扩展不得当作已有结论。

## 复核已有结果

安装依赖后，在仓库根目录执行：

```powershell
python verify_results.py
python verify_optimized.py
```

复核包括冻结模型哈希、需求种子隔离、逐车延误重算、信号安全时间及决策转移一致性。`results/optimized/comparison/verification.json` 保存交付时的复核结果。

本仓库不包含课程原始任务书，也不包含本机软件环境与登录信息。第三方软件按各自许可证使用。
