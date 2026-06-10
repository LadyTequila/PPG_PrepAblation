"""
C3: 信号归一化
==============

对窗口级特征张量 (n_channels, win_len) 做归一化。

mode:
    "none"        : 不做任何归一化（与 v2 等价；归一化交由训练侧
                    StandardScaler 处理）
    "robust_95"   : 每通道分别按"距均值最近 95% 数据"的 mean/std 做
                    z-score（DRIVEN 风格，从 gen_img_complete.m:15~22 port）
    "3sigma_clip" : 每通道分别做 3σ 裁剪 + z-score 标准化
                    （SleepPPG-Net2 风格）

注意：mode="none" 时输出 = 输入（zero-copy 返回），保证 v2 等价配置
下产出 pickle 与原 MESA_v2/delay10_win60_soft1/ byte-level 一致。
"""

import numpy as np


def apply(feat: np.ndarray, mode: str = "none") -> np.ndarray:
    """
    Parameters
    ----------
    feat : np.ndarray, shape (n_channels, win_len)
        PPG 多通道形态特征，已经过 ppg_extraction 与 resample。
    mode : str
        "none" / "robust_95" / "3sigma_clip"

    Returns
    -------
    np.ndarray, shape (n_channels, win_len), dtype float32
    """
    if mode == "none":
        return feat

    if mode == "robust_95":
        return _robust_95_zscore(feat)

    if mode == "3sigma_clip":
        return _3sigma_clip_zscore(feat)

    raise ValueError(f"Unknown normalization mode: {mode}")


# ── 实现 ─────────────────────────────────────────────────────────────────────

def _robust_95_zscore(feat: np.ndarray) -> np.ndarray:
    """
    DRIVEN 风格：从 gen_img_complete.m:15~22 直接 port。

    每个通道独立处理：
      1. 计算通道全数据的均值 m_tmp
      2. 取距 m_tmp 最近的 95% 数据点（剔除 5% 离群点）
      3. 在这 95% 数据上算 mean_95 / std_95
      4. (x - mean_95) / std_95
    """
    out = np.empty_like(feat, dtype=np.float32)
    eps = 1e-8

    for ch in range(feat.shape[0]):
        x = feat[ch].astype(np.float64)
        m_tmp = np.mean(x)
        dist = np.abs(x - m_tmp)
        sort_idx = np.argsort(dist)
        n95 = int(np.floor(0.95 * x.size))
        if n95 < 2:
            # 数据点太少，退化为标准 z-score
            mean_95 = float(np.mean(x))
            std_95 = float(np.std(x))
        else:
            kept = x[sort_idx[:n95]]
            mean_95 = float(np.mean(kept))
            std_95 = float(np.std(kept))
        out[ch] = ((x - mean_95) / (std_95 + eps)).astype(np.float32)

    return out


def _3sigma_clip_zscore(feat: np.ndarray) -> np.ndarray:
    """
    SleepPPG-Net2 风格：

    每个通道独立处理：
      1. 计算 mean / std
      2. 把数据 clip 到 [mean - 3·std, mean + 3·std]
      3. 用 clip 后数据的 mean/std 再次 z-score
    """
    out = np.empty_like(feat, dtype=np.float32)
    eps = 1e-8

    for ch in range(feat.shape[0]):
        x = feat[ch].astype(np.float64)
        mu = float(np.mean(x))
        sigma = float(np.std(x))
        # 3σ 裁剪
        lo = mu - 3.0 * sigma
        hi = mu + 3.0 * sigma
        x_clip = np.clip(x, lo, hi)
        # 再次 z-score
        mu2 = float(np.mean(x_clip))
        sigma2 = float(np.std(x_clip))
        out[ch] = ((x_clip - mu2) / (sigma2 + eps)).astype(np.float32)

    return out
