"""
C2: Hypopnea / Unsure 事件二次确认
==================================

返回经过确认规则过滤后的事件列表 events = [(start_sec, duration_sec), ...]

mode:
    "none"        : Apnea + Hypopnea + Unsure 全部直接保留（ApSense 原始）
    "adjacent"    : Apnea 直留; Hypop/Unsure 需"紧邻下一条 ScoredEvent 是 Arousal
                    或 SpO2 desaturation"（本组 v2）
    "flow_strict" : Apnea 直留; Hypop/Unsure 需 Flow 信号 ≥ 30% 下降 ≥ 10 秒
                    （本研究原创组件，参考 AASM 临床定义）
"""

import xml.etree.ElementTree as ET

import numpy as np

from . import flow_severity


def parse(
    xml_path: str,
    mode: str = "adjacent",
    flow: np.ndarray | None = None,
    fs_flow: float | None = None,
) -> tuple[list[tuple[float, float]], list[str], dict]:
    """
    Parameters
    ----------
    xml_path : str
        NSRR XML 标注路径
    mode : str
        "none" / "adjacent" / "flow_strict"
    flow : np.ndarray | None
        鼻气流信号；mode="flow_strict" 时必须提供
    fs_flow : float | None
        Flow 信号采样率；mode="flow_strict" 时必须提供

    Returns
    -------
    events : list[tuple[float, float]]
        经过确认规则过滤的呼吸事件 [(start_sec, dur_sec), ...]
    event_types : list[str]
        与 events 一一对应的事件类型字符串，取值范围
        {"ObstructiveApnea", "Hypopnea", "Unsure"}；C5=mode 模式需要
    stats : dict
        统计字典: obstructive_apnea / candidate_hypopnea_unsure /
                  confirmed / rejected / total_events
    """
    if mode == "flow_strict" and (flow is None or fs_flow is None):
        raise ValueError("mode='flow_strict' 需要提供 flow 与 fs_flow")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    scored_events = list(root.findall(".//ScoredEvent"))

    events: list[tuple[float, float]] = []
    event_types: list[str] = []
    n_apnea = 0
    n_candidate = 0
    n_confirmed = 0
    n_rejected = 0

    for i, ev in enumerate(scored_events):
        concept = (ev.findtext("EventConcept") or "").strip()
        c_lower = concept.lower()
        start = float(ev.findtext("Start") or 0)
        dur = float(ev.findtext("Duration") or 0)

        # Obstructive Apnea: 三种模式下都直接保留
        if "obstructive apnea" in c_lower:
            events.append((start, dur))
            event_types.append("ObstructiveApnea")
            n_apnea += 1
            continue

        # Hypopnea / Unsure 候选事件
        is_hypopnea = "hypopnea" in c_lower
        is_unsure = "unsure" in c_lower
        if is_hypopnea or is_unsure:
            n_candidate += 1
            ev_type = "Hypopnea" if is_hypopnea else "Unsure"

            if mode == "none":
                # 直接保留所有候选
                events.append((start, dur))
                event_types.append(ev_type)
                n_confirmed += 1
                continue

            if mode == "adjacent":
                # v2 原版：紧邻下一条 ScoredEvent 必须是 Arousal 或 SpO2 desat
                if i + 1 < len(scored_events):
                    nxt = scored_events[i + 1]
                    nxt_concept = (nxt.findtext("EventConcept") or "").strip().lower()
                    if "arousal" in nxt_concept or "spo2 desaturation" in nxt_concept:
                        events.append((start, dur))
                        event_types.append(ev_type)
                        n_confirmed += 1
                        continue
                n_rejected += 1
                continue

            if mode == "flow_strict":
                # 直接看 Flow 信号是否满足 AASM 阈值
                confirmed = flow_severity.is_hypopnea_confirmed_by_flow(
                    flow, fs_flow, start, dur,
                )
                if confirmed:
                    events.append((start, dur))
                    event_types.append(ev_type)
                    n_confirmed += 1
                else:
                    n_rejected += 1
                continue

            # 未知 mode
            raise ValueError(f"Unknown hypopnea_confirm mode: {mode}")

    stats = {
        "obstructive_apnea": n_apnea,
        "candidate_hypopnea_unsure": n_candidate,
        "confirmed": n_confirmed,
        "rejected": n_rejected,
        "total_events": n_apnea + n_confirmed,
    }
    return events, event_types, stats
