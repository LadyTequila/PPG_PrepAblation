"""
preprocess_modular.py —— 组件化预处理主入口
============================================

把毕设 prep_mesa_v2.py 的功能拆成 7 个可重组组件（C1~C7）的主入口；
每次调用通过参数指定一组组件配置，对应消融对比表的一行。

输出位置：
    {out_dir}/{experiment_id}/
        ├── mesa_fold{k}_x_train.pickle  (与 v2 同结构)
        ├── mesa_fold{k}_y_train.pickle
        ├── mesa_fold{k}_x_test.pickle
        ├── mesa_fold{k}_y_test.pickle
        ├── mesa_fold{k}_t_starts_train.pickle
        ├── mesa_fold{k}_t_starts_test.pickle
        ├── config.json          (本行实验的组件配置, 便于追溯)
        ├── fold_info.txt        (与 v2 同格式)
        └── subjects_index.json  (与 v2 同结构)

用法：
    # 命令行：跑 v2 等价配置（默认参数）
    python preprocess_modular.py --experiment-id row_02_v2_baseline

    # 命令行：跑 v2 + DRIVEN 归一化
    python preprocess_modular.py --normalization robust_95 \\
        --experiment-id row_03_driven_norm

    # 作为模块导入
    from preprocess_modular import preprocess_main
    preprocess_main(experiment_id="row_03_driven_norm",
                    normalization="robust_95")
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys
import warnings

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# 让 Python 找得到组件模块（脚本相对路径）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from preprocessing_components import trim  # noqa: E402
from preprocessing_components import confirm_hypopnea  # noqa: E402
from preprocessing_components import normalize as normalize_mod  # noqa: E402
from preprocessing_components import filter as filter_mod  # noqa: E402
from preprocessing_components import label as label_mod  # noqa: E402
from preprocessing_components import windowing  # noqa: E402

warnings.filterwarnings("ignore")


# ── 默认路径（保持与 v2 一致）─────────────────────────────────────────────────
DEFAULT_NPZ_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_raw_ppg_spo2_flow"
DEFAULT_XML_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_xml"
DEFAULT_OUT_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/ra_studies/MESA_ablation"

# ── 固定超参数（保持与 v2 一致）──────────────────────────────────────────────
MIN_SPO2      = 60
N_FOLDS       = 5
AHI_THRESHOLD = 15
RANDOM_SEED   = 42


# ── 单受试者处理 ──────────────────────────────────────────────────────────────

def process_subject_modular(npz_path: str,
                            xml_path: str,
                            *,
                            trim_strategy: str,
                            hypopnea_confirm: str,
                            normalization: str,
                            filter_mode: str,
                            label_mode: str,
                            delay_mode: str,
                            win_size: int,
                            stride: int,
                            ) -> dict | None:
    """
    处理单个受试者。返回 dict，结构与 v2 process_subject 兼容。
    无有效窗口时返回 None。
    """
    with np.load(npz_path, allow_pickle=True) as z:
        ppg = z["ppg"].astype("float32")
        spo2 = z["spo2"].astype("float32")
        fs_ppg = float(z["fs_ppg"][0])
        fs_spo2 = float(z["fs_spo2"][0])
        flow = z["flow"].astype("float32") if "flow" in z.files else None
        fs_flow = float(z["fs_flow"][0]) if "fs_flow" in z.files else None

    hz = int(round(fs_ppg))

    # 估算录制时长（fixed_30min/none 模式需要）
    recording_duration_sec = len(ppg) / fs_ppg

    # ── C1: 时窗裁剪 ──
    sleep_window = trim.get_window(
        xml_path, strategy=trim_strategy,
        recording_duration_sec=recording_duration_sec,
    )
    if sleep_window is None:
        return None
    onset, offset = sleep_window

    # ── C2: 事件确认 ──
    events, event_types, event_stats = confirm_hypopnea.parse(
        xml_path, mode=hypopnea_confirm,
        flow=flow if hypopnea_confirm == "flow_strict" else None,
        fs_flow=fs_flow if hypopnea_confirm == "flow_strict" else None,
    )

    # AHI（基于裁剪后的睡眠时长）
    sleep_dur_sec = offset - onset
    ahi = (event_stats["total_events"] / (sleep_dur_sec / 3600.0)) if sleep_dur_sec > 0 else 0.0

    features_list = []
    labels_list = []
    t_starts_list = []

    for t_start in windowing.iterate(onset, offset, win_size, stride):
        # SpO2 质量过滤
        spo2_start = int(t_start * fs_spo2)
        spo2_end = int((t_start + win_size) * fs_spo2)
        spo2_win = spo2[spo2_start:spo2_end]
        if len(spo2_win) == 0 or np.min(spo2_win) < MIN_SPO2:
            continue

        # PPG 窗口
        ppg_start = int(t_start * fs_ppg)
        ppg_end = int((t_start + win_size) * fs_ppg)
        ppg_win = ppg[ppg_start:ppg_end]
        if len(ppg_win) < win_size * hz:
            continue

        # C4: 滤波 + 形态特征提取
        # C3: 归一化（在 extraction 之后立即应用）
        try:
            feat = filter_mod.ppg_extraction(
                ppg_win, hz=hz, win_size=win_size,
                filter_mode=filter_mode,
            )
            feat = normalize_mod.apply(feat, mode=normalization)
        except Exception:
            continue

        # C5 + C6: 标签
        labels = label_mod.make_array(
            events, t_start, win_size,
            label_mode=label_mode,
            delay_mode=delay_mode,
            event_types=event_types,
        )

        features_list.append(feat)
        labels_list.append(labels)
        t_starts_list.append(int(t_start))

    return {
        "features": features_list,
        "labels": labels_list,
        "t_starts": t_starts_list,
        "ahi": ahi,
        "sleep_window": (int(onset), int(offset)),
        "events": [(float(es), float(ed)) for es, ed in events],
        "event_stats": event_stats,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def preprocess_main(*,
                    npz_dir: str = DEFAULT_NPZ_DIR,
                    xml_dir: str = DEFAULT_XML_DIR,
                    out_dir: str = DEFAULT_OUT_DIR,
                    experiment_id: str = "row_02_v2_baseline",
                    # 组件参数（默认对应"本组 v2 完整 = 毕设 v2_delay0"，组件编码 b-b-a-a-a-a-a）
                    trim_strategy: str = "xml",
                    hypopnea_confirm: str = "adjacent",
                    normalization: str = "none",
                    filter_mode: str = "cheby_hp_20hz",
                    label_mode: str = "hard",
                    delay_mode: str = "none",
                    windowing_spec="v2_60s_50p",
                    # 评估参数
                    n_folds: int = N_FOLDS,
                    random_seed: int = RANDOM_SEED,
                    ahi_threshold: float = AHI_THRESHOLD,
                    ) -> None:
    """主入口：跑一次组件化预处理，输出到 {out_dir}/{experiment_id}/。"""
    win_size, stride = windowing.resolve(windowing_spec)
    exp_dir = os.path.join(out_dir, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    # 记录本次实验的组件配置
    config = {
        "experiment_id": experiment_id,
        "components": {
            "C1_trim_strategy": trim_strategy,
            "C2_hypopnea_confirm": hypopnea_confirm,
            "C3_normalization": normalization,
            "C4_filter_mode": filter_mode,
            "C5_label_mode": label_mode,
            "C6_delay_mode": delay_mode,
            "C7_windowing": {
                "spec": windowing_spec if isinstance(windowing_spec, str) else "custom",
                "win_size": win_size,
                "stride": stride,
            },
        },
        "evaluation": {
            "n_folds": n_folds,
            "random_seed": random_seed,
            "ahi_threshold": ahi_threshold,
        },
    }
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"实验 ID: {experiment_id}")
    print(f"组件配置:")
    for k, v in config["components"].items():
        print(f"  {k:30s} = {v}")
    print(f"输出目录: {exp_dir}\n")

    # ── 遍历受试者 ──
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "mesa_*_raw_ppg_spo2_flow.npz")))
    if not npz_files:
        raise FileNotFoundError(f"在 {npz_dir} 下未找到任何 NPZ 文件")
    print(f"找到 {len(npz_files)} 个受试者")

    all_features = []
    all_labels = []
    all_t_starts = []
    all_sleep_windows = []
    all_events = []
    all_ahi = []
    subject_ids = []

    total_apnea = 0
    total_candidate = 0
    total_confirmed = 0
    total_rejected = 0
    skipped_no_sleep = 0

    for npz_path in tqdm(npz_files, desc="处理受试者"):
        match = re.search(r"mesa_(\d{4})_raw", os.path.basename(npz_path))
        if not match:
            print(f"  警告：跳过无法解析编号的文件 {npz_path}")
            continue
        subj_id = match.group(1)

        xml_path = os.path.join(xml_dir, f"mesa-sleep-{subj_id}-nsrr.xml")
        if not os.path.exists(xml_path):
            print(f"  警告：受试者 {subj_id} 缺少 XML 标注文件，跳过")
            continue

        result = process_subject_modular(
            npz_path, xml_path,
            trim_strategy=trim_strategy,
            hypopnea_confirm=hypopnea_confirm,
            normalization=normalization,
            filter_mode=filter_mode,
            label_mode=label_mode,
            delay_mode=delay_mode,
            win_size=win_size,
            stride=stride,
        )

        if result is None:
            print(f"  警告：受试者 {subj_id} 无睡眠分期数据，跳过")
            skipped_no_sleep += 1
            continue

        feats = result["features"]
        if len(feats) == 0:
            print(f"  警告：受试者 {subj_id} 无有效窗口，跳过")
            continue

        # 累计统计
        s = result["event_stats"]
        total_apnea     += s.get("obstructive_apnea", 0)
        total_candidate += s.get("candidate_hypopnea_unsure", 0)
        total_confirmed += s.get("confirmed", 0)
        total_rejected  += s.get("rejected", 0)

        onset, offset = result["sleep_window"]
        ahi = result["ahi"]
        tqdm.write(
            f"  {subj_id}: 睡眠 {onset}s~{offset}s "
            f"({(offset-onset)/3600:.1f}h) | "
            f"事件: apnea={s['obstructive_apnea']} "
            f"hyp确认={s['confirmed']}/{s['candidate_hypopnea_unsure']} | "
            f"窗口={len(feats)} | AHI={ahi:.1f}"
        )

        all_features.append(np.array(feats,  dtype=np.float32))
        all_labels.append(np.array(result["labels"], dtype=np.float32))
        all_t_starts.append(np.array(result["t_starts"], dtype=np.int64))
        all_sleep_windows.append(result["sleep_window"])
        all_events.append(result["events"])
        all_ahi.append(ahi)
        subject_ids.append(subj_id)

    n_subjects = len(subject_ids)
    print(f"\n{'='*60}")
    print(f"有效受试者：{n_subjects} 人（{skipped_no_sleep} 人因无睡眠分期被跳过）")
    print(f"事件统计:")
    print(f"  Obstructive apnea:                    {total_apnea}")
    print(f"  Hypopnea/Unsure 候选:                {total_candidate}")
    print(f"    → 确认: {total_confirmed}  / 拒绝: {total_rejected}")
    print(f"  总有效事件: {total_apnea + total_confirmed}")
    print(f"{'='*60}")

    if n_subjects == 0:
        raise RuntimeError("没有任何有效受试者，请检查数据路径和文件格式")

    # ── 5 折 AHI 分层 CV ──
    ahi_arr = np.array(all_ahi)
    ahi_severe = (ahi_arr > ahi_threshold).astype(int)
    subject_idx = np.arange(n_subjects)
    print(f"AHI 分布 — severe (>{ahi_threshold}): {ahi_severe.sum()}  "
          f"non-severe: {(1-ahi_severe).sum()}")

    skf = StratifiedShuffleSplit(n_splits=n_folds, random_state=random_seed,
                                  test_size=0.10)

    log_lines = [
        f"preprocess_modular.py 预处理日志\n",
        f"experiment_id: {experiment_id}\n",
        f"组件配置: {config['components']}\n",
        f"\n5-Fold Train/Test subject indices\n",
    ]
    fold_assignments = []

    fold = 0
    for train_idx, test_idx in skf.split(subject_idx, ahi_severe):
        print(f"\nFold {fold}  train={len(train_idx)} subjects  test={len(test_idx)} subjects")
        log_lines.append(
            f"Fold {fold}\n  TRAIN: {[subject_ids[i] for i in train_idx]}\n"
            f"  TEST:  {[subject_ids[i] for i in test_idx]}\n"
        )

        x_train = [all_features[i] for i in train_idx]
        y_train = [all_labels[i]   for i in train_idx]
        x_test  = [all_features[i] for i in test_idx]
        y_test  = [all_labels[i]   for i in test_idx]
        t_starts_train = [all_t_starts[i] for i in train_idx]
        t_starts_test  = [all_t_starts[i] for i in test_idx]

        prefix = os.path.join(exp_dir, f"mesa_fold{fold}")
        with open(prefix + "_x_train.pickle", "wb") as f:
            pickle.dump(x_train, f, protocol=4)
        with open(prefix + "_y_train.pickle", "wb") as f:
            pickle.dump(y_train, f, protocol=4)
        with open(prefix + "_x_test.pickle", "wb") as f:
            pickle.dump(x_test, f, protocol=4)
        with open(prefix + "_y_test.pickle", "wb") as f:
            pickle.dump(y_test, f, protocol=4)
        with open(prefix + "_t_starts_train.pickle", "wb") as f:
            pickle.dump(t_starts_train, f, protocol=4)
        with open(prefix + "_t_starts_test.pickle", "wb") as f:
            pickle.dump(t_starts_test, f, protocol=4)

        fold_assignments.append({
            "fold": fold,
            "train_idx": [int(i) for i in train_idx],
            "test_idx":  [int(i) for i in test_idx],
            "train_subjects": [subject_ids[i] for i in train_idx],
            "test_subjects":  [subject_ids[i] for i in test_idx],
        })

        total_train_win = sum(x.shape[0] for x in x_train)
        total_test_win = sum(x.shape[0] for x in x_test)
        print(f"       train windows: {total_train_win}  test windows: {total_test_win}")
        fold += 1

    log_path = os.path.join(exp_dir, "fold_info.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    # ── subjects_index.json ──
    subjects_index_path = os.path.join(exp_dir, "subjects_index.json")
    with open(subjects_index_path, "w", encoding="utf-8") as f:
        json.dump({
            "subject_ids": subject_ids,
            "ahi": [float(a) for a in all_ahi],
            "sleep_windows": [list(sw) for sw in all_sleep_windows],
            "events": all_events,
            "n_windows_per_subject": [int(len(t)) for t in all_t_starts],
            "fold_assignments": fold_assignments,
            "config": {
                **config["components"],
                "n_folds": n_folds,
                "random_seed": random_seed,
                "ahi_threshold": ahi_threshold,
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"\n完成！pickle 文件保存至 {exp_dir}")
    print(f"折次信息: {log_path}")
    print(f"受试者元信息: {subjects_index_path}")
    print(f"实验配置: {os.path.join(exp_dir, 'config.json')}")


# ── 命令行入口 ────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="组件化预处理主入口（消融实验用）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR)
    p.add_argument("--xml-dir", default=DEFAULT_XML_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--experiment-id", default="row_02_v2_baseline",
                   help="本次实验的标识，决定输出子目录名")

    # 组件参数
    p.add_argument("--trim-strategy", default="xml",
                   choices=["fixed_30min", "xml", "none"])
    p.add_argument("--hypopnea-confirm", default="adjacent",
                   choices=["none", "adjacent", "flow_strict"])
    p.add_argument("--normalization", default="none",
                   choices=["none", "robust_95", "3sigma_clip"])
    p.add_argument("--filter-mode", default="cheby_hp_20hz",
                   choices=["cheby_hp_20hz", "cheby_lp_8hz"])
    p.add_argument("--label-mode", default="hard",
                   choices=["hard", "soft", "mode"])
    p.add_argument("--delay-mode", default="none",
                   choices=["none", "single_10", "multi"])
    p.add_argument("--windowing-spec", default="v2_60s_50p",
                   choices=list(windowing.PRESETS.keys()))

    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    p.add_argument("--ahi-threshold", type=float, default=AHI_THRESHOLD)
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    preprocess_main(
        npz_dir=args.npz_dir,
        xml_dir=args.xml_dir,
        out_dir=args.out_dir,
        experiment_id=args.experiment_id,
        trim_strategy=args.trim_strategy,
        hypopnea_confirm=args.hypopnea_confirm,
        normalization=args.normalization,
        filter_mode=args.filter_mode,
        label_mode=args.label_mode,
        delay_mode=args.delay_mode,
        windowing_spec=args.windowing_spec,
        n_folds=args.n_folds,
        random_seed=args.random_seed,
        ahi_threshold=args.ahi_threshold,
    )
