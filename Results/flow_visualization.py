"""
Flow 严格 AASM 判定的可视化抽样验证
================================================================

为论文 row_8 caveat 服务——回答这样一个问题：
    "Flow 严格判定拒绝 79% 的 Hypopnea 候选，是不是太严了？"

通过可视化抽样：
    - 随机挑 K 个被 Flow 判定**拒绝**的候选 Hypopnea
    - 随机挑 K 个被 Flow 判定**确认**的候选 Hypopnea
    - 画出 |Flow| 信号 + AASM 30% 下降阈值线 + 事件区间
    - 让读者肉眼判断"拒绝是否合理"

预期结论
--------
若 rejected 事件的 |Flow| 信号在事件区间内**没有明显下降**，则确认拒绝合理；
即 AUROC 提升 +6.34 主要来自"标签纯度提升"而非"任务变易"。

用法
----
    python flow_visualization.py

输出
----
    ra_studies/Results/flow_visualization/
    ├── rejected_examples.png   (K 个被拒绝事件的网格图)
    ├── confirmed_examples.png  (K 个被确认事件的网格图)
    └── summary.md              (索引 + 解读说明)
"""

from __future__ import annotations

import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = [
    "SimHei"
]
plt.rcParams["axes.unicode_minus"] = False


# 项目根目录 / 路径
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent.parent          # ApSense-main/
_RA = _THIS.parent.parent                    # ra_studies/

# 让组件模块可导入
sys.path.insert(0, str(_RA))
from preprocessing_components import flow_severity  # noqa: E402

# ── 数据路径（同 preprocess_modular.py 默认值；Windows 写法）──────────────────
DEFAULT_NPZ_DIR = str(_ROOT / "shared_subset" / "mesa_quality7_raw_ppg_spo2_flow")
DEFAULT_XML_DIR = str(_ROOT / "shared_subset" / "mesa_quality7_xml")
DEFAULT_OUT_DIR = str(_THIS.parent / "flow_visualization")

# ── 抽样参数 ─────────────────────────────────────────────────────────────────
N_SUBJECTS_TO_SCAN = 30      # 扫多少个 subject 收集候选
K_REJECTED = 6               # 画 K 个被拒绝的事件
K_CONFIRMED = 6              # 画 K 个被确认的事件
RANDOM_SEED = 42

# ── 可视化窗口参数 ────────────────────────────────────────────────────────────
CONTEXT_BEFORE_SEC = 60.0    # 事件起始前显示多少秒（基线区间的一部分）
CONTEXT_AFTER_SEC = 20.0     # 事件结束后显示多少秒
# Flow 严格判定的参数（与 flow_severity 默认值一致）
PRE_EVENT_BASELINE_SEC = 120.0
DROP_RATIO = 0.30
MIN_DURATION_SEC = 10.0


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def list_subject_ids(npz_dir: str) -> list[str]:
    """从 npz_dir 列出所有受试者 4 位编号。"""
    ids = []
    for fname in sorted(os.listdir(npz_dir)):
        if fname.startswith("mesa_") and fname.endswith("_raw_ppg_spo2_flow.npz"):
            # mesa_0012_raw_ppg_spo2_flow.npz → "0012"
            sid = fname.split("_")[1]
            ids.append(sid)
    return ids


def load_subject(npz_dir: str, xml_dir: str, sid: str):
    """加载一个受试者的 Flow 信号和 XML 标注。"""
    npz_path = os.path.join(npz_dir, f"mesa_{sid}_raw_ppg_spo2_flow.npz")
    xml_path = os.path.join(xml_dir, f"mesa-sleep-{sid}-nsrr.xml")

    if not os.path.exists(npz_path) or not os.path.exists(xml_path):
        return None

    with np.load(npz_path, allow_pickle=True) as z:
        if "flow" not in z.files:
            return None
        flow = z["flow"].astype("float32")
        fs_flow = float(z["fs_flow"][0])

    return {"sid": sid, "flow": flow, "fs_flow": fs_flow, "xml_path": xml_path}


def extract_hypopnea_candidates(xml_path: str) -> list[tuple[float, float, str]]:
    """从 XML 拉出所有 Hypopnea / Unsure 候选事件。
    返回 [(start_sec, duration_sec, type_str), ...]
    """
    candidates = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for ev in root.findall(".//ScoredEvent"):
            concept = (ev.findtext("EventConcept") or "").strip()
            c_lower = concept.lower()
            if "hypopnea" in c_lower or "unsure" in c_lower:
                start = float(ev.findtext("Start") or 0)
                dur = float(ev.findtext("Duration") or 0)
                ev_type = "Hypopnea" if "hypopnea" in c_lower else "Unsure"
                candidates.append((start, dur, ev_type))
    except Exception as e:
        print(f"  ⚠️  XML 解析失败 {xml_path}: {e}")
    return candidates


def evaluate_candidate(subj: dict, ev_start: float, ev_dur: float) -> dict:
    """对单个候选事件计算 Flow 判定相关的数字。"""
    flow = subj["flow"]
    fs_flow = subj["fs_flow"]

    # 基线计算（与 flow_severity 内部完全一致）
    baseline_start = max(0.0, ev_start - PRE_EVENT_BASELINE_SEC)
    baseline_end = ev_start
    bs = int(baseline_start * fs_flow)
    be = int(baseline_end * fs_flow)
    if be <= bs or (baseline_end - baseline_start) < 30.0:
        return {"confirmed": False, "baseline_amp": None, "threshold": None,
                "reason": "无足够 pre-event 基线（事件接近录制起始）"}

    baseline_amp = float(np.median(np.abs(flow[bs:be])))
    if baseline_amp <= 1e-8:
        return {"confirmed": False, "baseline_amp": 0.0, "threshold": 0.0,
                "reason": "基线信号接近 0（可能传感器脱落）"}

    threshold = baseline_amp * (1.0 - DROP_RATIO)

    confirmed = flow_severity.is_hypopnea_confirmed_by_flow(
        flow, fs_flow, ev_start, ev_dur,
    )

    return {
        "confirmed": confirmed,
        "baseline_amp": baseline_amp,
        "threshold": threshold,
        "reason": "通过 AASM ≥30% / ≥10s 判定" if confirmed else "未达到 AASM ≥30% / ≥10s",
    }


def plot_event(ax, subj: dict, ev_start: float, ev_dur: float, ev_type: str, eval_result: dict):
    """在给定的 axis 上画一个事件的可视化。"""
    flow = subj["flow"]
    fs_flow = subj["fs_flow"]

    # 显示窗口范围
    show_start_sec = max(0.0, ev_start - CONTEXT_BEFORE_SEC)
    show_end_sec = min(len(flow) / fs_flow, ev_start + ev_dur + CONTEXT_AFTER_SEC)
    ss = int(show_start_sec * fs_flow)
    se = int(show_end_sec * fs_flow)

    # |Flow| 信号（绝对值更便于看下降幅度）
    flow_segment = np.abs(flow[ss:se])
    t = np.arange(len(flow_segment)) / fs_flow + show_start_sec

    ax.plot(t, flow_segment, color="#1f77b4", linewidth=0.5, alpha=0.7, label="|Flow|")

    # 1 秒滑动均值（更直观体现包络）
    win = int(fs_flow)
    if len(flow_segment) > win:
        smooth = np.convolve(flow_segment, np.ones(win) / win, mode="same")
        ax.plot(t, smooth, color="#1f77b4", linewidth=1.5, label="1s 滑均")

    # 基线
    if eval_result["baseline_amp"] is not None:
        ax.axhline(
            eval_result["baseline_amp"], color="#2ca02c", linestyle="--",
            linewidth=1.5, label=f"基线 (pre-event 中位数) = {eval_result['baseline_amp']:.3f}",
        )

    # 30% 下降阈值
    if eval_result["threshold"] is not None:
        ax.axhline(
            eval_result["threshold"], color="#d62728", linestyle=":",
            linewidth=1.5, label=f"AASM 阈值 (70% 基线) = {eval_result['threshold']:.3f}",
        )

    # 事件区间红色阴影
    ax.axvspan(ev_start, ev_start + ev_dur, alpha=0.15, color="red", label=f"NSRR 标注事件 ({ev_type})")

    # 标题
    status = "Confirmed" if eval_result["confirmed"] else "Rejected"
    color = "green" if eval_result["confirmed"] else "red"
    ax.set_title(
        f"Subject {subj['sid']} | t = {ev_start:.0f}s, dur = {ev_dur:.0f}s | {status}",
        color=color, fontsize=10,
    )
    ax.set_xlabel("时间 (秒)", fontsize=8)
    ax.set_ylabel("|Flow|", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.grid(alpha=0.3)
    ax.set_xlim(show_start_sec, show_end_sec)


def make_grid_figure(events_list: list[dict], title: str, out_path: str):
    """把 K 个事件画成 2×3 或 3×2 网格图。"""
    n = len(events_list)
    if n == 0:
        print(f"  ⚠️  无事件可画，跳过 {title}")
        return

    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), dpi=120)
    axes = np.atleast_2d(axes).flatten()

    for i, ev in enumerate(events_list):
        plot_event(
            axes[i],
            ev["subj"], ev["ev_start"], ev["ev_dur"], ev["ev_type"],
            ev["eval_result"],
        )

    # 多余 axes 隐藏
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  保存 {out_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)

    print(f"NPZ 目录:  {DEFAULT_NPZ_DIR}")
    print(f"XML 目录:  {DEFAULT_XML_DIR}")
    print(f"输出目录:  {DEFAULT_OUT_DIR}")
    print()

    all_subjects = list_subject_ids(DEFAULT_NPZ_DIR)
    if not all_subjects:
        print(f"❌ 未找到任何 npz 文件，请检查路径: {DEFAULT_NPZ_DIR}")
        sys.exit(1)
    print(f"共找到 {len(all_subjects)} 个受试者；随机抽 {N_SUBJECTS_TO_SCAN} 个扫描候选事件")

    chosen_subjects = random.sample(all_subjects, min(N_SUBJECTS_TO_SCAN, len(all_subjects)))

    rejected_pool = []
    confirmed_pool = []

    for i, sid in enumerate(chosen_subjects):
        subj = load_subject(DEFAULT_NPZ_DIR, DEFAULT_XML_DIR, sid)
        if subj is None:
            print(f"  [{i+1}/{N_SUBJECTS_TO_SCAN}] {sid}: 加载失败，跳过")
            continue

        candidates = extract_hypopnea_candidates(subj["xml_path"])
        n_rej = n_conf = 0
        for (ev_start, ev_dur, ev_type) in candidates:
            res = evaluate_candidate(subj, ev_start, ev_dur)
            entry = {
                "subj": subj, "ev_start": ev_start, "ev_dur": ev_dur,
                "ev_type": ev_type, "eval_result": res,
            }
            if res["confirmed"]:
                confirmed_pool.append(entry)
                n_conf += 1
            else:
                rejected_pool.append(entry)
                n_rej += 1
        print(f"  [{i+1}/{N_SUBJECTS_TO_SCAN}] subject {sid}: "
              f"候选 {len(candidates)} 个 → 确认 {n_conf} / 拒绝 {n_rej}")

    print()
    print(f"汇总: 共收集 {len(confirmed_pool)} 个 confirmed + {len(rejected_pool)} 个 rejected")

    if len(rejected_pool) < K_REJECTED:
        print(f"⚠️  rejected 数量不足 {K_REJECTED}（只有 {len(rejected_pool)}），将全用")
    if len(confirmed_pool) < K_CONFIRMED:
        print(f"⚠️  confirmed 数量不足 {K_CONFIRMED}（只有 {len(confirmed_pool)}），将全用")

    # 随机抽样
    rejected_sample = random.sample(rejected_pool, min(K_REJECTED, len(rejected_pool)))
    confirmed_sample = random.sample(confirmed_pool, min(K_CONFIRMED, len(confirmed_pool)))

    # 画图
    print()
    print("正在生成可视化...")
    make_grid_figure(
        rejected_sample,
        title=f"Rejected Examples — Flow 严格 AASM 判定拒绝的候选 Hypopnea 事件 (n={len(rejected_sample)})",
        out_path=os.path.join(DEFAULT_OUT_DIR, "rejected_examples.png"),
    )
    make_grid_figure(
        confirmed_sample,
        title=f"Confirmed Examples — Flow 严格 AASM 判定确认的候选 Hypopnea 事件 (n={len(confirmed_sample)})",
        out_path=os.path.join(DEFAULT_OUT_DIR, "confirmed_examples.png"),
    )

    # 写一份 summary.md
    n_total = len(confirmed_pool) + len(rejected_pool)
    confirm_rate = len(confirmed_pool) / n_total if n_total > 0 else 0
    summary_path = os.path.join(DEFAULT_OUT_DIR, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Flow 严格判定可视化抽样验证\n\n")
        f.write(f"**为论文 row_8 caveat 服务**——验证 AUROC 提升 +6.34 主要来自标签纯度提升而非任务变易。\n\n")
        f.write(f"## 抽样统计\n\n")
        f.write(f"- 扫描 subject 数: {N_SUBJECTS_TO_SCAN}\n")
        f.write(f"- 总候选事件数 (Hypopnea + Unsure): {n_total}\n")
        f.write(f"- 被 Flow 确认: {len(confirmed_pool)} ({confirm_rate:.1%})\n")
        f.write(f"- 被 Flow 拒绝: {len(rejected_pool)} ({1-confirm_rate:.1%})\n\n")
        f.write(f"## 抽样可视化\n\n")
        f.write(f"### 1. 被拒绝事件 ({len(rejected_sample)} 个)\n\n")
        f.write(f"![rejected_examples](rejected_examples.png)\n\n")
        f.write(f"**判读要点**：\n")
        f.write(f"- 红色阴影 = NSRR 标注的事件区间\n")
        f.write(f"- 绿色虚线 = 事件前 120s 的 \\|Flow\\| 中位数（AASM 基线）\n")
        f.write(f"- 红色点线 = 基线的 70%（30% 下降阈值）\n")
        f.write(f"- 蓝色细线 = 原始 \\|Flow\\| 信号，粗线 = 1 秒滑均\n")
        f.write(f"- **若事件区间内的滑均信号在阈值线以上，则确实没有 AASM 定义的"
                f"30% 下降——拒绝合理**\n\n")
        f.write(f"### 2. 被确认事件 ({len(confirmed_sample)} 个)\n\n")
        f.write(f"![confirmed_examples](confirmed_examples.png)\n\n")
        f.write(f"**判读要点**：事件区间内的滑均应**明显跌破红色阈值线**且持续 ≥ 10 秒。\n\n")
        f.write(f"## 解读\n\n")
        f.write(f"若 rejected 组的可视化显示这些事件在事件区间内的 \\|Flow\\| 未明显下降，\n")
        f.write(f"则证明 Flow 严格判定**确实剔除了「标注边界模糊的低质量样本」**，\n")
        f.write(f"AUROC 提升 +6.34 的主要来源是**标签纯度提升**，而非「任务本身变易」。\n\n")
        f.write(f"## 实验参数\n\n")
        f.write(f"- pre_event_baseline_sec = {PRE_EVENT_BASELINE_SEC}s\n")
        f.write(f"- drop_ratio = {DROP_RATIO} (即 30%)\n")
        f.write(f"- min_duration_sec = {MIN_DURATION_SEC}s\n")
        f.write(f"- tolerance_sec = ±5s\n")
        f.write(f"- random_seed = {RANDOM_SEED}\n")
    print(f"  ✓  保存 {summary_path}")

    print()
    print("完成！可视化结果已保存到 ra_studies/Results/flow_visualization/")


if __name__ == "__main__":
    main()
