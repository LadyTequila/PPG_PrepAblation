"""
C5 + C6: 标签生成（粒度 + 延迟）
================================

label_mode :
    "hard" : 硬二值标签，逐秒 0/1（ApSense 原始）
    "soft" : 软标签，逐秒填充事件覆盖比例 ratio ∈ [0, 1]（本组 v2）
    "mode" : 事件类型 mode（DRIVEN 风格，整窗填充窗口内事件类型的众数）

delay_mode :
    "none"      : delay = 0
    "single_10" : delay = 10 秒（本组 v2 默认）
    "multi"     : 对 [0, 10, 15, 20, 25] 五个延迟生成 5 份标签，取均值
                  得到"delay-aware 软标签"（每秒覆盖比例 ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}），
                  shape (win_size,)。自动触发 v2 现有 soft 训练分支

Notes
-----
v2 原版 make_label_array() 的等价配置:
    make_array(events, t_start, win_size,
               label_mode="soft", delay_mode="single_10")

注意 label_mode="mode" 需要传入 event_types 参数（每个事件的类型字符串），
否则会退化为按 events 数量 mode（即都标 1）。
"""

import numpy as np


DEFAULT_MULTI_DELAYS = (0, 10, 15, 20, 25)


def make_array(
    events: list[tuple[float, float]],
    window_start_sec: int,
    win_size: int,
    label_mode: str = "soft",
    delay_mode: str = "single_10",
    event_types: list[str] | None = None,
    multi_delays: tuple[int, ...] = DEFAULT_MULTI_DELAYS,
) -> np.ndarray:
    """
    生成窗口标签。

    Returns
    -------
    np.ndarray, dtype float32, shape = (win_size,)
      - none / single_10 模式：与 v2 原版完全兼容
      - multi 模式：5 个 delay 的标签 mean 聚合后的 delay-aware 软标签
        （数值在 [0, 0.2, 0.4, 0.6, 0.8, 1.0] 内取离散值）
    """
    if delay_mode == "none":
        delays = [0]
    elif delay_mode == "single_10":
        delays = [10]
    elif delay_mode == "multi":
        delays = list(multi_delays)
    else:
        raise ValueError(f"Unknown delay_mode: {delay_mode}")

    arrs = []
    for d in delays:
        arr = _make_single_delay_array(
            events, window_start_sec, win_size, d,
            label_mode=label_mode, event_types=event_types,
        )
        arrs.append(arr)

    if delay_mode == "multi":
        # 方案 A：mean 聚合 —— 5 个 delay 下的标签平均成 delay-aware 软标签
        # 物理含义：每秒位置上 5 个 delay 中覆盖该秒的比例
        # 工程效果：shape 与 single delay 一致 (win_size,)，
        # 数值落在 [0, 1] 区间，自动让毕设 runner is_soft=True 走 soft 训练分支
        return np.mean(arrs, axis=0).astype(np.float32)
    return arrs[0]


# ── 单一延迟的标签生成 ───────────────────────────────────────────────────────

def _make_single_delay_array(
    events: list[tuple[float, float]],
    window_start_sec: int,
    win_size: int,
    delay_sec: int,
    label_mode: str,
    event_types: list[str] | None,
) -> np.ndarray:
    """
    返回 shape (win_size,) 的标签。

    label_mode:
        "hard" : 逐秒 0/1
        "soft" : 整段填充 ratio
        "mode" : 整段填充窗口内事件类型的众数（int）
    """
    # 步骤 1：得到逐秒二值占据数组（与 v2 原版 make_label_array 的循环一致）
    binary = np.zeros(win_size, dtype=np.float32)
    occupying_event_indices: list[int] = []  # 落在本窗口的事件下标（用于 mode 模式）

    for ev_idx, (ev_start, ev_dur) in enumerate(events):
        shifted_start = ev_start + delay_sec
        shifted_end = shifted_start + ev_dur
        overlapping = False
        for s in range(win_size):
            t = window_start_sec + s
            if shifted_start <= t < shifted_end:
                binary[s] = 1.0
                overlapping = True
        if overlapping:
            occupying_event_indices.append(ev_idx)

    if label_mode == "hard":
        return binary

    if label_mode == "soft":
        # 与 v2 原版完全一致：填充事件秒占比
        ratio = float(binary.sum()) / win_size
        return np.full(win_size, ratio, dtype=np.float32)

    if label_mode == "mode":
        return _mode_label(
            win_size, binary, occupying_event_indices, event_types,
        )

    raise ValueError(f"Unknown label_mode: {label_mode}")


def _mode_label(
    win_size: int,
    binary: np.ndarray,
    occupying_event_indices: list[int],
    event_types: list[str] | None,
) -> np.ndarray:
    """
    DRIVEN 风格的事件类型 mode 标签：

      - 如果窗口内无事件覆盖, label = 0
      - 否则在窗口内覆盖的事件类型中取众数（mode），整段填充该众数对应的整数类标
      - event_types 为 None 时退化为"有事件 = 1, 无事件 = 0"（等价 hard）
    """
    if binary.sum() == 0:
        return np.zeros(win_size, dtype=np.float32)

    if event_types is None or len(event_types) == 0:
        return np.ones(win_size, dtype=np.float32)

    # 收集落在本窗口内的事件类型
    occupied_types = [
        event_types[i] for i in occupying_event_indices if i < len(event_types)
    ]
    if not occupied_types:
        return np.zeros(win_size, dtype=np.float32)

    # 用字符串到整数的映射作为类标
    # (本组的常见事件类型: ObstructiveApnea / Hypopnea / Unsure)
    label_map = {
        "ObstructiveApnea": 1.0,
        "Hypopnea": 2.0,
        "Unsure": 3.0,
    }
    int_labels = [label_map.get(t, 0.0) for t in occupied_types]

    # 取众数
    vals, counts = np.unique(int_labels, return_counts=True)
    mode_val = float(vals[np.argmax(counts)])
    return np.full(win_size, mode_val, dtype=np.float32)
