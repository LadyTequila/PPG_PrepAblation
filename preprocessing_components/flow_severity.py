"""
Flow 信号下降严格度计算工具
============================

用于 C2 (c) flow_strict 模式：
判断一个候选事件区间内的鼻气流是否真的 ≥ 30% 下降持续 ≥ 10 秒。

实现严格遵循 AASM 临床定义：
    - 基线幅值: 事件起始前 120 秒（pre-event reference baseline）的 |Flow| 中位数
                ——不包含事件本身，避免事件期幅值压低基线导致自我对齐
    - 容忍窗口: 候选事件区间 ± 5 秒（允许 NSRR 标注的轻微偏差）
    - 持续条件: 该容忍窗口内 ≥ 10 秒连续低于基线 70%
    - 边界处理: 事件起始距录制开始 < 30 秒时无法构造可信基线 → 保守拒绝
"""

import numpy as np


def is_hypopnea_confirmed_by_flow(
    flow: np.ndarray,
    fs_flow: float,
    ev_start_sec: float,
    ev_duration_sec: float,
    *,
    pre_event_baseline_sec: float = 120.0,
    min_baseline_sec: float = 30.0,
    tolerance_sec: float = 5.0,
    drop_ratio: float = 0.30,
    min_duration_sec: float = 10.0,
) -> bool:
    """
    检查 Flow 信号是否满足 AASM Hypopnea 判定：
    "气流幅值下降 ≥ drop_ratio 持续 ≥ min_duration_sec 秒"。

    基线遵循 AASM 临床定义——取事件前 pre_event_baseline_sec 秒的正常呼吸幅值，
    避免把事件本身纳入基线计算导致的自我对齐误差。

    Parameters
    ----------
    flow : np.ndarray
        整段鼻气流信号
    fs_flow : float
        Flow 信号采样率
    ev_start_sec, ev_duration_sec : float
        候选 Hypopnea 事件的起始秒和持续时长
    pre_event_baseline_sec : float
        基线窗口长度（事件起始前的多少秒，默认 120s，对应 AASM 推荐的
        2 分钟 pre-event reference baseline）
    min_baseline_sec : float
        基线窗口最低有效长度。如事件起始距离录制开始不足此值，
        无法构造可信基线 → 返回 False（保守拒绝）
    tolerance_sec : float
        候选事件起止边界各向外延伸的容忍秒数（默认 ±5 秒，
        允许 NSRR 标注的轻微偏差）
    drop_ratio : float
        AASM 阈值（默认 0.30，即下降 30% 以上视为 Hypopnea）
    min_duration_sec : float
        最短持续时长（默认 10 秒）

    Returns
    -------
    True 表示该候选事件被 Flow 信号确认为有效 Hypopnea。
    """
    if flow is None or len(flow) == 0 or fs_flow <= 0:
        return False

    # 1. 基线幅值：取事件起始前 pre_event_baseline_sec 秒的 |Flow| 中位数
    #    （AASM 临床定义；不把事件本身包含进来，避免自我对齐压低基线）
    baseline_start_sec = max(0.0, ev_start_sec - pre_event_baseline_sec)
    baseline_end_sec = ev_start_sec

    # 录制起点附近的事件：pre-event 不够长 → 无法构造可信基线，保守拒绝
    if baseline_end_sec - baseline_start_sec < min_baseline_sec:
        return False

    bs = int(baseline_start_sec * fs_flow)
    be = int(baseline_end_sec * fs_flow)
    if be <= bs:
        return False

    baseline_segment = np.abs(flow[bs:be])
    baseline_amp = float(np.median(baseline_segment))
    if baseline_amp <= 1e-8:
        return False  # 基线本身已接近零，无法判定相对下降

    # 2. 在容忍窗口内计算"逐秒幅值"：对每秒取绝对值的均值（近似呼吸幅值包络）
    win_start_sec = max(0.0, ev_start_sec - tolerance_sec)
    win_end_sec = min(len(flow) / fs_flow, ev_start_sec + ev_duration_sec + tolerance_sec)

    n_seconds = int(np.floor(win_end_sec - win_start_sec))
    if n_seconds <= 0:
        return False

    per_sec_amp = np.zeros(n_seconds, dtype=np.float64)
    for i in range(n_seconds):
        s = int((win_start_sec + i) * fs_flow)
        e = int((win_start_sec + i + 1) * fs_flow)
        if e <= s:
            continue
        per_sec_amp[i] = np.mean(np.abs(flow[s:e]))

    # 3. 判定每一秒是否"低于基线 (1 - drop_ratio) 倍"
    threshold = baseline_amp * (1.0 - drop_ratio)
    is_low_per_sec = per_sec_amp < threshold

    # 4. 找出最长连续低区间，判断是否 ≥ min_duration_sec 秒
    max_run = _max_consecutive_true(is_low_per_sec)
    return max_run >= min_duration_sec


def _max_consecutive_true(arr: np.ndarray) -> int:
    """计算布尔数组中最长 True 连续段的长度。"""
    if len(arr) == 0:
        return 0
    max_run = 0
    cur_run = 0
    for v in arr:
        if v:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    return max_run
