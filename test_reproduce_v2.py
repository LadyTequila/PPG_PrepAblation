"""
验证脚本：组件化版本能否在 v2 等价配置下逐字节复现毕设原 pickle
=====================================================================

为什么不直接验证 "v2 baseline" (= v2_delay0)
--------------------------------------------
按本研究的口径，"本组 v2 完整 = 毕设 v2_delay0"，对应组件编码 b-b-a-a-a-a-a；
但本地 MESA_v2/ 下**没有** delay0_win60_soft0 这个目录（只有 delay10_win60_soft1）。
因此 byte-level 验证若想"现在就能跑"，就只能拿本地已有的产物作为对照。

本脚本默认验证以下"v2 等价配置"（毕设 v2_delay10_soft1，组件编码 b-b-a-a-b-b-a）:
    trim_strategy    = "xml"
    hypopnea_confirm = "adjacent"
    normalization    = "none"
    filter_mode      = "cheby_hp_20hz"
    label_mode       = "soft"        # b
    delay_mode       = "single_10"   # b
    windowing_spec   = "v2_60s_50p"  (win_size=60, stride=30)

→ 与毕设产物 MESA_v2/delay10_win60_soft1/ 做逐字节比对。

这等价于"验证组件化的实现没改变 C5/C6 切换后 v2 路径的输出"——
一旦通过，就能从同一份代码"无失真"切换出 v2_delay0 (b-b-a-a-a-a-a)。

通过标准
--------
5 折 × 2 split × 3 类 pickle = 30 个文件全部 SHA256 一致。
hash 不一致但 numpy 数组逐元素相等记为"近似等价"。

用法
----
    # 1. 先跑组件化版本的"delay10_soft1 配置"
    python preprocess_modular.py --label-mode soft --delay-mode single_10 \
        --experiment-id row_02c_v2_delay10_soft1

    # 2. 跑本验证脚本（默认对比 row_02c_v2_delay10_soft1 vs delay10_win60_soft1）
    python test_reproduce_v2.py

    # 可选：切换到其他对比目标（若已生成 v2_delay0 产物）
    python test_reproduce_v2.py --modular-id row_02_v2_baseline \
        --v2-dir C:/Users/.../MESA_v2/delay0_win60_soft0
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys

import numpy as np


DEFAULT_V2_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/MESA_v2/delay10_win60_soft1"
DEFAULT_MODULAR_ID = "row_02c_v2_delay10_soft1"
DEFAULT_OUT_ROOT = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/ra_studies/MESA_ablation"

N_FOLDS = 5
SPLITS = ["train", "test"]
ARRAY_NAMES = ["x", "y", "t_starts"]


# ── 比对工具 ─────────────────────────────────────────────────────────────────

def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_list_of_arrays(old, new) -> tuple[bool, str]:
    if not isinstance(old, list) or not isinstance(new, list):
        return False, f"不是 list: type(old)={type(old).__name__}, type(new)={type(new).__name__}"
    if len(old) != len(new):
        return False, f"list 长度不等: {len(old)} vs {len(new)}"
    for i, (a, b) in enumerate(zip(old, new)):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            return False, f"item [{i}] shape 不同: {a.shape} vs {b.shape}"
        if a.dtype != b.dtype:
            return False, f"item [{i}] dtype 不同: {a.dtype} vs {b.dtype}"
        if not np.array_equal(a, b):
            max_abs_diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            if max_abs_diff == 0.0:
                continue
            return False, f"item [{i}] 不相等（max abs diff = {max_abs_diff}）"
    return True, "list of arrays 完全一致"


# ── 主流程 ────────────────────────────────────────────────────────────────────

def run_comparison(v2_dir: str, modular_dir: str) -> int:
    print(f"{'='*78}")
    print(f"对比目标:")
    print(f"  原版:    {v2_dir}")
    print(f"  组件版:  {modular_dir}")
    print(f"{'='*78}\n")

    if not os.path.isdir(v2_dir):
        print(f"❌ 原版目录不存在: {v2_dir}")
        print("    若想验证 v2_delay0 (=本组 v2 完整, b-b-a-a-a-a-a)，请先用毕设默认参数生成:")
        print("        python prep_mesa_v2.py        # 产出 MESA_v2/delay0_win60_soft0/")
        sys.exit(1)
    if not os.path.isdir(modular_dir):
        print(f"❌ 组件版目录不存在: {modular_dir}")
        print("    先跑:")
        print(f"        python preprocess_modular.py --label-mode soft --delay-mode single_10 \\")
        print(f"            --experiment-id {os.path.basename(modular_dir)}")
        sys.exit(1)

    n_total = 0
    n_hash_equal = 0
    n_logically_equal = 0
    n_failed = 0

    for fold in range(N_FOLDS):
        for split in SPLITS:
            for name in ARRAY_NAMES:
                n_total += 1
                fname = f"mesa_fold{fold}_{name}_{split}.pickle"
                old_path = os.path.join(v2_dir, fname)
                new_path = os.path.join(modular_dir, fname)

                if not os.path.exists(old_path):
                    print(f"⚠️ {fname}  原版缺失，跳过")
                    continue
                if not os.path.exists(new_path):
                    print(f"❌ {fname}  组件版缺失")
                    n_failed += 1
                    continue

                old_hash = hash_file(old_path)
                new_hash = hash_file(new_path)

                if old_hash == new_hash:
                    n_hash_equal += 1
                    n_logically_equal += 1
                    print(f"✅ {fname}  hash 一致")
                    continue

                with open(old_path, "rb") as f:
                    old = pickle.load(f)
                with open(new_path, "rb") as f:
                    new = pickle.load(f)
                ok, msg = compare_list_of_arrays(old, new)
                if ok:
                    n_logically_equal += 1
                    print(f"⚠️ {fname}  hash 不同但数组逐元素一致 — {msg}")
                else:
                    n_failed += 1
                    print(f"❌ {fname}  数组不一致 — {msg}")

    print(f"\n{'='*78}")
    print(f"汇总:")
    print(f"  总文件数:                {n_total}")
    print(f"  hash 一致:               {n_hash_equal}")
    print(f"  逻辑一致(含 hash 一致):  {n_logically_equal}")
    print(f"  失败:                    {n_failed}")
    print(f"{'='*78}")

    if n_failed == 0 and n_logically_equal == n_total:
        if n_hash_equal == n_total:
            print(f"\n🎉 完美：所有 {n_total} 个 pickle 文件 byte-level 一致")
        else:
            print(f"\n✅ 验证通过（{n_hash_equal} byte 级一致 + "
                  f"{n_logically_equal - n_hash_equal} 数组逐元素一致）")
    else:
        print(f"\n❌ 验证失败 — 有 {n_failed} 个 pickle 与原版不一致")
        sys.exit(2)

    return 0


def main():
    p = argparse.ArgumentParser(
        description="byte-level 验证组件化预处理是否复现毕设 v2 pickle"
    )
    p.add_argument("--v2-dir", default=DEFAULT_V2_DIR,
                   help="毕设产物路径（含 mesa_foldK_*.pickle 30 个）")
    p.add_argument("--modular-id", default=DEFAULT_MODULAR_ID,
                   help="组件化产物的 experiment_id，对应子目录名")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                   help="组件化产物的根目录")
    args = p.parse_args()

    modular_dir = os.path.join(args.out_root, args.modular_id)
    return run_comparison(args.v2_dir, modular_dir)


if __name__ == "__main__":
    main()
