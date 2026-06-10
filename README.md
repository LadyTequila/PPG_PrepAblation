# PPG-OSA 检测中各家预处理方法的组件化消融对比

本仓库是一项研究助理阶段的工作：把 ApSense、DRIVEN、SleepPPG-Net2 等代表性工作的数据预处理拆成可独立替换的组件，在统一的下游模型与评估协议下逐项替换，量化每个预处理组件对 PPG 单模态睡眠呼吸暂停（OSA）检测任务的边际贡献。研究的核心发现是：**最有价值的提升来自本工作原创的"基于鼻气流的严格 AASM 事件判定"（AUROC 87.19%，相对基准提升 6.34 个百分点），而多数从其他论文直接借鉴的预处理组件在本任务下无效甚至有害——这本身验证了"预处理方法是为特定任务和数据量身定做、跨任务照搬需谨慎"的方法学观点。**

本工作是毕业设计[「基于 PPG 信号的睡眠呼吸暂停事件检测」](https://github.com/LadyTequila/PPG_DSepMIL)的延续，沿用了毕设的 v2 预处理流水线与 DSepMIL 模型作为基础。

## 研究动机

不同论文的 PPG/PSG 预处理流水线各有差异，但直接做"整条流水线对整条流水线"的端到端对比并不可行也不公平：各家的输入维度、通道选择、任务定义、下游模型都不同，比出来的差异分不清到底来自预处理还是来自模型。为此本工作采用组件化消融（受控变量法）——把预处理切成 7 个相对独立的环节，每次只替换其中一个，其余全部保持本组毕设 v2 的做法不变，从而把性能差异精确归因到被替换的那个环节。

需要强调的是，这与 ApSense 原文将 DRIVEN 等作为 baseline 的对比是**两个正交的研究问题**：ApSense 把各家网络架构强行适配到自己的数据流水线下，比的是"网络架构层面"的拟合能力；本工作保持网络与数据统一、只换预处理组件，比的是"预处理方法层面"的有效性。

## 七个预处理组件

| 编号 | 组件 | 候选做法 |
|---|---|---|
| C1 | 时窗裁剪 | 固定 30 分钟裁剪 / XML 精确裁剪 / 不裁剪 |
| C2 | 事件二次确认 | 不确认 / 紧邻 Arousal-Desat 确认 / **Flow 严格 AASM 判定（本工作原创）** |
| C3 | 信号归一化 | 训练侧 StandardScaler / DRIVEN 95% 分位 / SleepPPG-Net2 3σ 裁剪 |
| C4 | 滤波 | Chebyshev II 高通 20Hz / Chebyshev II 低通 8Hz |
| C5 | 标签粒度 | 硬二值 / 软标签 / DRIVEN 事件类型 mode |
| C6 | 延迟对齐 | 无延迟 / 固定单延迟 / **多延迟 mean 聚合（本工作扩展）** |
| C7 | 窗口与步长 | 60s/30s / 30s/5s 高密度短窗 |

各组件的原理详见 `各组件原理.md`，实验设计与方法学讨论详见 `研究计划_预处理对比消融.md`。

## 主要结果

在 MESA quality=7 子集（276 受试者）、DSepMIL 下游模型、5 折交叉验证、统一 thres=0 的设定下，单组件消融的关键结果（相对基准 row_2 的 AUROC 80.85%）：

| 组件配置 | 来源 | AUROC | Pearson r | 相对基准 |
|---|---|---|---|---|
| Flow 严格 AASM 判定 | 本工作原创 | **87.19 ± 1.69** | 0.8034 | **+6.34** |
| 多延迟 mean 聚合 | 本工作扩展 | **84.21 ± 1.65** | 0.7956 | **+3.36** |
| 基准（本组 v2 完整） | 毕设 v2 | 80.85 ± 0.96 | 0.7668 | — |
| DRIVEN 95% 归一化 | DRIVEN | 79.74 ± 1.08 | 0.6844 | -1.11 |
| SleepPPG-Net2 滤波+归一化 | SleepPPG-Net2 | 79.65 ± 1.18 | 0.6791 | -1.20 |
| DRIVEN 短窗 30s/5s | DRIVEN | 80.21（单折） | 0.5139 | AUROC 持平但 AHI 估计骤降 |
| DRIVEN 三组件组合 | DRIVEN | 76.56（单折） | 0.7546 | -4.29（协同负面） |
| DRIVEN mode 标签 | DRIVEN | 训练崩溃 | — | 与二分类 BCE 不兼容 |

完整指标表与详细分析见 `Results/results_summary.md`（及其 PDF 版本）。

### 几个核心发现

- **Flow 严格判定**直接调取鼻气流信号、按 AASM 临床定义核实每个低通气事件（事件前 2 分钟基线、气流下降 ≥30% 持续 ≥10s），把特异度从 76.60% 提升到 95.44%；这一组件之所以可行，是因为所用 MESA 子集恰好保留了 Flow 通道。
- **多延迟 mean 聚合**把毕设的延迟对齐从单一延迟扩展到多延迟平均，自然产生了类似软标签的平滑效果，单组件即接近毕设需要软标签加延迟才能达到的最优。
- **归一化的"时机"比"方法"更重要**：两种借鉴来的离线归一化都不及毕设训练侧的在线归一化。
- **组件之间存在协同负面效应**：DRIVEN 三个组件单独尚可，组合后反而显著低于基准，反驳了"好组件可随意叠加"的直觉。
- **两类失败模式**：mode 标签与二分类损失不兼容导致训练崩溃（显性失败）；含 mode 的早期组合训练表面正常但标签数值越界、结果不可信（隐性失败）。这警示跨论文借鉴时"能跑通"不等于"结果可信"。

## 项目结构

```
ra_studies/
├── README.md                          本文件
├── 研究计划_预处理对比消融.md          研究背景、方法学论证、实验设计、消融矩阵
├── 重构方案.md                        组件化重构的设计决策记录
├── 各组件原理.md                      面向汇报的组件原理通俗说明
│
├── preprocess_modular.py              组件化预处理主入口（按参数组合调用各组件）
├── test_reproduce_v2.py               byte-level 验证：组件版能否无损复现毕设 v2 产物
├── diag_row_05_shape.py               多延迟标签的张量形状诊断脚本
│
├── preprocessing_components/          7 个可独立替换的预处理组件
│   ├── trim.py                        C1 时窗裁剪
│   ├── confirm_hypopnea.py            C2 事件二次确认
│   ├── flow_severity.py               C2 的 Flow 严格 AASM 判定（本工作原创）
│   ├── normalize.py                   C3 信号归一化
│   ├── filter.py                      C4 滤波 + PPG 形态特征提取
│   ├── label.py                       C5+C6 标签粒度与延迟对齐
│   └── windowing.py                   C7 窗口与步长
│
├── MESA_ablation/                     各 row 的预处理 pickle 产物（不入库）
└── Results/                           实验结果与可视化
    ├── results_summary.md / .pdf      完整结果汇总与分析
    ├── flow_visualization.py          Flow 严格判定的可视化抽样验证脚本
    └── logs/                          各 row 的训练与评估日志（不入库）
```

## 与毕设代码的依赖关系

本仓库只覆盖**预处理组件化**部分。**训练与评估**复用毕设的 `model_v2/` 框架（基于 PyTorch + Hydra 的 DSepMIL 训练流程），本仓库不重复实现。完整跑通一行消融实验的流程是：

1. 用本仓库的 `preprocess_modular.py` 按指定组件配置生成 pickle 产物
2. 用毕设 `model_v2/main.py`（指定 `model=DSepMIL`、`dataset_dir` 指向上一步产物）训练
3. 用毕设 `model_v2/evaluate.py` 评估

毕设代码与本仓库的关系详见毕设主仓库 README。

## 快速开始

### 1. 生成某一行消融的预处理产物

`preprocess_modular.py` 通过命令行参数指定 7 个组件的配置，默认对应"本组 v2 完整"基准。例如：

```bash
# 基准：本组 v2 完整（XML 裁剪 + 紧邻确认 + 训练侧归一化 + 硬标签 + 无延迟 + 60s 窗）
python preprocess_modular.py --experiment-id row_02_v2_baseline

# Flow 严格 AASM 判定（本工作原创组件，效果最优）
python preprocess_modular.py --hypopnea-confirm flow_strict --experiment-id row_08_flow_strict

# 多延迟 mean 聚合
python preprocess_modular.py --delay-mode multi --experiment-id row_05_multi_delay

# DRIVEN 95% 分位归一化
python preprocess_modular.py --normalization robust_95 --experiment-id row_03_driven_norm

# SleepPPG-Net2 风格（低通滤波 + 3σ 归一化）
python preprocess_modular.py --normalization 3sigma_clip --filter-mode cheby_lp_8hz --experiment-id row_04_sleepppgnet2_norm
```

每次运行会在 `MESA_ablation/{experiment-id}/` 下输出 5 折切分的 pickle 文件，并附 `config.json` 记录本行使用的组件配置（评估时会被自动读取以适配窗口/步长等参数）。

### 2. 验证组件化重构无损

重构后的组件版在 v2 等价配置下应能逐字节复现毕设原产物：

```bash
# 先用组件版跑 v2 等价配置，再做 byte-level 比对
python preprocess_modular.py --label-mode soft --delay-mode single_10 --experiment-id row_02c_v2_delay10_soft1
python test_reproduce_v2.py
```

### 3. Flow 严格判定的可视化验证

抽样若干被严格判定拒绝/确认的低通气事件，画出气流信号与 AASM 阈值线，验证拒绝是否合理：

```bash
python Results/flow_visualization.py
```

## 数据与环境

### 数据访问

本研究使用 [MESA Sleep Study](https://sleepdata.org/datasets/mesa) 数据集，需通过 [NSRR](https://sleepdata.org/) 注册并提交研究计划申请后获取。**本仓库不包含任何原始 PSG 数据，也不包含预处理后的中间产物**——MESA 受数据使用协议约束不可二次分发，而 `MESA_ablation/` 下的 pickle 产物体积庞大且可由原始数据完整重建，因此均已通过 `.gitignore` 排除。

所用子集为 MESA 中 PPG 信号质量评分 = 7 的 276 名受试者，每人保留 PPG、SpO₂、Nasal Flow 三个通道。其中 Flow 通道是本工作原创组件 C2（Flow 严格 AASM 判定）的基础——并非所有 PPG 子集都保留气流信号，本研究恰好可用。获得 NSRR 授权后，需先将 EDF 转为 npz（仅保留三个通道）并按如下命名放置，才能运行本仓库的预处理脚本：

```
shared_subset/mesa_quality7_raw_ppg_spo2_flow/mesa_XXXX_raw_ppg_spo2_flow.npz
shared_subset/mesa_quality7_xml/mesa-sleep-XXXX-nsrr.xml
```

数据准备的详细步骤与受试者编号清单可参考毕设主仓库 [PPG_DSepMIL](https://github.com/LadyTequila/PPG_DSepMIL)，两个仓库使用完全相同的数据子集与 5 折划分。

### 环境

环境依赖与毕设主仓库一致（PyTorch、numpy、scipy、scikit-learn、neurokit2、pyedflib 等）。本仓库的预处理部分只依赖 numpy/scipy/neurokit2，**不需要 GPU**；训练与评估复用毕设 `model_v2/` 框架，需要 GPU。

## 状态与后续方向

当前完成了 7 行单组件消融与 1 行组合实验；其中短窗与组合两行因训练耗时较长目前为单折结果，补齐为 5 折是首要后续工作。更长期的方向包括：基于已保留的三通道做 PPG+SpO₂+Flow 多模态融合；以及把已验证有效的 Flow 严格判定推广到跨数据集泛化评估。详见 `研究计划_预处理对比消融.md` 第六节。
