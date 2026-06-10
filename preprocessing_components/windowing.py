"""
C7: 滑窗参数与迭代器
====================

预设两个常用 (win_size, stride) 配置：
    "v2_60s_50p" : win_size=60s, stride=30s（50% 重叠，本组 v2 默认）
    "driven_30s" : win_size=30s, stride=5s（DRIVEN 风格高密度短窗）

用户也可以直接传 (win_size, stride) 元组，不限于预设。
"""

from typing import Iterator


# 预设配置
PRESETS = {
    "v2_60s_50p": (60, 30),
    "driven_30s": (30, 5),
}


def resolve(spec) -> tuple[int, int]:
    """
    把用户给定的窗口配置解析为 (win_size, stride)。

    spec 可以是：
      - 字符串 "v2_60s_50p" / "driven_30s"（按 PRESETS 查表）
      - tuple (win_size, stride)
    """
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"Unknown windowing preset: {spec}")
        return PRESETS[spec]
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return int(spec[0]), int(spec[1])
    raise ValueError(f"Cannot resolve windowing spec: {spec}")


def iterate(sleep_onset: float,
            sleep_offset: float,
            win_size: int,
            stride: int) -> Iterator[int]:
    """
    生成窗口起始秒 t_start 序列。

    与 v2 原版 process_subject 的 range(t_start_min, t_start_max, stride) 完全一致。
    """
    t_start_min = int(sleep_onset)
    t_start_max = int(sleep_offset) - win_size
    for t_start in range(t_start_min, t_start_max, stride):
        yield t_start
