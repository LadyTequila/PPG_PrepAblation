"""
C1: 睡眠时窗裁剪
================

返回 (sleep_onset_sec, sleep_offset_sec)，决定主流程的滑窗起止边界。

strategy:
    "fixed_30min" : 固定首尾各去 30 分钟（ApSense / DRIVEN 风格）
    "xml"         : 从 XML 睡眠分期事件中提取真正的睡眠时窗（本组 v2）
    "none"        : 不裁剪，使用整段录制时长
"""

import xml.etree.ElementTree as ET

import numpy as np


def get_window(xml_path: str,
               strategy: str = "xml",
               recording_duration_sec: float | None = None,
               ) -> tuple[float, float] | None:
    """
    根据 strategy 返回 (sleep_onset, sleep_offset)。

    Parameters
    ----------
    xml_path : str
        NSRR XML 标注路径
    strategy : str
        "fixed_30min" / "xml" / "none"
    recording_duration_sec : float | None
        整段录制时长（"fixed_30min" 和 "none" 模式需要）。
        如果 None，会尝试从 XML 中估计（取最后一个事件的结束时刻）。

    Returns
    -------
    (onset, offset) 或 None（XML 模式下无睡眠分期数据时）
    """
    if strategy == "xml":
        return _get_window_xml(xml_path)

    if recording_duration_sec is None:
        recording_duration_sec = _estimate_recording_duration(xml_path)

    if strategy == "fixed_30min":
        return _get_window_fixed_30min(recording_duration_sec)

    if strategy == "none":
        return (0.0, float(recording_duration_sec))

    raise ValueError(f"Unknown trim strategy: {strategy}")


# ── 实现 ─────────────────────────────────────────────────────────────────────

def _get_window_xml(xml_path: str) -> tuple[float, float] | None:
    """
    v2 原版逻辑：遍历 Stages|Stages 事件，
    取第一个非 Wake 分期的 Start 与最后一个非 Wake 分期的结束时刻。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sleep_onset = None
    sleep_offset = None

    for ev in root.iter("ScoredEvent"):
        event_type = (ev.findtext("EventType") or "").strip().lower()
        concept = (ev.findtext("EventConcept") or "").strip().lower()

        if "stages" not in event_type:
            continue

        if "wake" in concept or concept.endswith("|0"):
            continue

        start = float(ev.findtext("Start") or 0)
        dur = float(ev.findtext("Duration") or 0)
        end = start + dur

        if sleep_onset is None:
            sleep_onset = start
        sleep_offset = end

    if sleep_onset is not None and sleep_offset is not None:
        return (float(sleep_onset), float(sleep_offset))
    return None


def _get_window_fixed_30min(recording_duration_sec: float) -> tuple[float, float]:
    """
    ApSense / DRIVEN 风格：首尾各去 30 分钟。
    若总时长不足 60 分钟，则等比例对称裁剪 10%。
    """
    cut = 30 * 60  # 30 分钟
    if recording_duration_sec <= 2 * cut:
        # 时长不足，对称裁掉 10%
        cut = recording_duration_sec * 0.1
    onset = cut
    offset = recording_duration_sec - cut
    return (float(onset), float(offset))


def _estimate_recording_duration(xml_path: str) -> float:
    """
    估计整段录制时长（取所有 ScoredEvent 中 Start+Duration 的最大值）。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    max_end = 0.0
    for ev in root.iter("ScoredEvent"):
        start = float(ev.findtext("Start") or 0)
        dur = float(ev.findtext("Duration") or 0)
        end = start + dur
        if end > max_end:
            max_end = end
    return max_end
