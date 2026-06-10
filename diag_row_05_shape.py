"""
一次性诊断脚本：检查 row_05_multi_delay 的 pickle shape
========================================================

目的
----
确认 C6=multi delay 配置下 label.make_array 产出的标签是否符合 (n_win, 5, 60) 的预期，
以及毕设训练框架 model_v2/runner.py 能否 swallow 这种形状。

判断三种情况
------------
情况 1: y[0].shape == (n_win, 60)
    → label.py multi delay 实现没真正起作用（bug）

情况 2: y[0].shape == (n_win, 5, 60)
    → 预期 shape；vstack 后 (N_total, 5, 60)；毕设训练框架几乎一定不能直接 swallow，
      需要决定 reduce 策略或改训练框架

情况 3: y[0].shape 各 subject 不一致 / vstack 失败
    → 边界处理 bug，需查 label.py

用法
----
    python diag_row_05_shape.py
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np


ROW = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/ra_studies/MESA_ablation/row_05_multi_delay"


def main() -> int:
    if not os.path.isdir(ROW):
        print(f"❌ row_05_multi_delay 产物目录不存在: {ROW}")
        print("    请先运行:")
        print("        python preprocess_modular.py --delay-mode multi --experiment-id row_05_multi_delay")
        return 1

    y_path = os.path.join(ROW, "mesa_fold0_y_train.pickle")
    x_path = os.path.join(ROW, "mesa_fold0_x_train.pickle")

    for p in (y_path, x_path):
        if not os.path.exists(p):
            print(f"❌ 缺失文件: {p}")
            return 1

    # ── y (标签) 诊断 ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("  y_train (标签) 诊断")
    print("=" * 70)
    with open(y_path, "rb") as f:
        y = pickle.load(f)

    print(f"type(y)                   = {type(y).__name__}")
    print(f"len(y)  (n_subjects)      = {len(y)}")
    print(f"type(y[0])                = {type(y[0]).__name__}")
    print(f"y[0].shape                = {y[0].shape}")
    print(f"y[0].dtype                = {y[0].dtype}")
    print(f"y[0].min(), y[0].max()    = {float(y[0].min()):.4f}, {float(y[0].max()):.4f}")
    print(f"y[0].mean()               = {float(y[0].mean()):.4f}")

    # 各 subject shape 一致性检查
    shapes = [tuple(arr.shape) for arr in y]
    unique_window_dims = set(s[1:] for s in shapes)  # 除了 n_win 维之外，其它维度应该一致
    print(f"unique non-window dims    = {unique_window_dims}")
    if len(unique_window_dims) > 1:
        print("⚠️  各 subject 的非窗口维度不一致 —— 这是 bug")
    else:
        print("✓  各 subject 非窗口维度一致")

    # vstack 测试
    try:
        y_stacked = np.vstack(y)
        print(f"vstack(y).shape           = {y_stacked.shape}")
        print(f"vstack(y).dtype           = {y_stacked.dtype}")
    except Exception as e:
        print(f"❌ vstack 失败: {e}")
        y_stacked = None

    # ── x (特征) 诊断（参照基准）─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  x_train (特征) 诊断（参照基准）")
    print("=" * 70)
    with open(x_path, "rb") as f:
        x = pickle.load(f)
    print(f"x[0].shape  (n_win, C, T) = {x[0].shape}")
    print(f"x[0].dtype                = {x[0].dtype}")
    try:
        x_stacked = np.vstack(x)
        print(f"vstack(x).shape           = {x_stacked.shape}")
    except Exception as e:
        print(f"❌ vstack(x) 失败: {e}")

    # ── 模拟毕设训练框架对 y 的处理 ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  模拟 model_v2/runner.py 对 y 的关键操作")
    print("=" * 70)
    if y_stacked is None:
        print("⚠️  vstack 已失败，无法继续模拟训练框架")
        return 2

    # 复刻 runner.py:382-383 的处理逻辑
    try:
        all_y_raw = np.array([iny for outy in y for iny in outy])
        print(f"runner.py:383 all_y_raw.shape = {all_y_raw.shape}")
    except Exception as e:
        print(f"❌ runner.py:383 模拟失败: {e}")

    # 复刻 runner.py:386 的软标签检测
    try:
        is_soft = bool(np.any((y_stacked > 0) & (y_stacked < 1)))
        print(f"runner.py:386 is_soft         = {is_soft}")
    except Exception as e:
        print(f"❌ is_soft 检测失败: {e}")

    # 复刻 runner.py:390 软标签的训练目标转化
    try:
        all_y_mean = y_stacked.mean(axis=1).astype(np.float32)
        print(f"runner.py:390 mean(axis=1).shape = {all_y_mean.shape}")
        print(f"  → 预期 1D (N_total,);")
        if all_y_mean.ndim == 1:
            print("  ✓  shape 匹配，soft label 训练分支可走")
        else:
            print(f"  ❌  得到 {all_y_mean.ndim}D，毕设训练框架会算错")
    except Exception as e:
        print(f"❌ mean(axis=1) 失败: {e}")

    print()
    print("=" * 70)
    print("  诊断完成 —— 把以上输出贴回去由 Claude 判断走哪条路线")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
