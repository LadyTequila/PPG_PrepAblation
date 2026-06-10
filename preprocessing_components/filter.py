"""
C4: 滤波 + PPG 形态特征提取
============================

filter_mode 决定 ppg_extraction() 内部使用的滤波器：

    "cheby_hp_20hz" : Chebyshev II 型 2 阶高通，截止 20 Hz（v2 原版）
    "cheby_lp_8hz"  : Chebyshev II 型 8 阶低通，截止 8 Hz（SleepPPG-Net2 风格）

ppg_extraction() 接口与 v2 完全兼容，返回形状为 (7, RESAMPLE_N) 的特征矩阵。
"""

import neurokit2 as nk
import numpy as np
from scipy import signal
from scipy.signal import argrelextrema


RESAMPLE_N = 60  # 与 v2 一致


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """与原 prep_mesa_v2.moving_average 完全等价。"""
    return np.convolve(x, np.ones(w), "valid") / w


def _build_filter_sos(filter_mode: str, hz: int):
    """根据 filter_mode 构造 SOS 滤波器系数。"""
    if filter_mode == "cheby_hp_20hz":
        # v2 原版：2 阶 Chebyshev II 高通，截止 20Hz，rs=0.1 dB
        return signal.cheby2(2, 0.1, 20, "hp", fs=hz, output="sos")
    if filter_mode == "cheby_lp_8hz":
        # SleepPPG-Net2 风格：8 阶 Chebyshev II 低通，截止 8Hz，40 dB 衰减
        return signal.cheby2(8, 40, 8, "lp", fs=hz, output="sos")
    raise ValueError(f"Unknown filter_mode: {filter_mode}")


def ppg_extraction(raw_ppg: np.ndarray,
                   hz: int,
                   win_size: int,
                   filter_mode: str = "cheby_hp_20hz") -> np.ndarray:
    """
    从一段原始 PPG 信号提取 7 个形态特征。

    Parameters
    ----------
    raw_ppg : np.ndarray
        原始 PPG 时域信号
    hz : int
        采样率
    win_size : int
        窗口大小（秒，仅用于参数语义；实际采样数 = win_size * hz）
    filter_mode : str
        "cheby_hp_20hz" / "cheby_lp_8hz"

    Returns
    -------
    np.ndarray, shape (7, RESAMPLE_N), dtype float32
        7 通道形态特征 [PWA, SPD, DPD, PA, PPI, dPWA, dPPI]，
        经 scipy.signal.resample 重采样到固定长度 RESAMPLE_N。

    Notes
    -----
    本函数的 filter_mode="cheby_hp_20hz" 路径与 v2 原版的
    prep_mesa_v2.ppg_extraction() 完全等价。
    """
    # 1. nk 清洗 + 主峰检测（filter_mode 无关）
    ppg_clean = nk.ppg_clean(raw_ppg, sampling_rate=hz)
    info = nk.ppg_findpeaks(ppg_clean, sampling_rate=hz)
    peaks = info["PPG_Peaks"]
    rr_interval = (np.diff(peaks) / hz) * 1000  # ms

    # 2. 自定义滤波（filter_mode 决定具体滤波器与方向）
    sos = _build_filter_sos(filter_mode, hz)
    if filter_mode == "cheby_lp_8hz":
        # SleepPPG-Net2 风格：零相位双向滤波。
        # 参考: DavyWJW/sleep-staging-models extract_mesa_data.py:142
        # Behar 课题组 SleepPPG-Net 体系一致使用 sosfiltfilt 消除滤波器相位延迟
        filtered = signal.sosfiltfilt(sos, raw_ppg)
    else:
        # v2 原版必须保留单向 sosfilt，否则破坏与
        # MESA_v2/delay10_win60_soft1/ 的 byte-level 一致性
        filtered = signal.sosfilt(sos, raw_ppg)
    filtered_ma = moving_average(filtered, hz // 2)

    # 3. 谷值与二次峰检测
    local_minima = argrelextrema(filtered_ma, np.less)[0]
    local_maxima = argrelextrema(filtered_ma, np.greater)[0]

    rm_index, rm2_index = [], []
    diffs = np.diff(local_minima)
    for k in range(len(diffs)):
        if diffs[k] < 30:
            rm_index.append(k)
            rm2_index.append(k + 1)
    sel_min = np.delete(local_minima, rm_index)
    sel_max = np.delete(local_maxima, rm2_index)

    # 4. 计算 PWA / SPD / DPD / PA
    pwa_list, systole_list, diastole_list, area_list = [], [], [], []

    count_min = 0
    for mini in sel_min:
        count_min += 1
        count_max = 0
        for maxi in sel_max:
            count_max += 1
            if maxi > mini:
                pwa_list.append(float(filtered_ma[maxi] - filtered_ma[mini]))
                systole_list.append(float(maxi - mini))
                if count_min + 1 < len(sel_min):
                    diastole_list.append(
                        float(sel_min[count_min + 1] - sel_max[count_max - 1])
                    )
                area_list.append(float(np.sum(filtered_ma[mini:maxi])))
                break

    # 5. 等长重采样到 RESAMPLE_N
    pwa_arr      = signal.resample(pwa_list, RESAMPLE_N)
    systole_arr  = signal.resample(systole_list, RESAMPLE_N)
    diastole_arr = signal.resample(diastole_list, RESAMPLE_N)
    area_arr     = signal.resample(area_list, RESAMPLE_N)
    rr_arr       = signal.resample(rr_interval, RESAMPLE_N)
    diff_rr_arr  = signal.resample(np.diff(rr_interval), RESAMPLE_N)
    diff_pwa_arr = signal.resample(np.diff(pwa_list), RESAMPLE_N)

    return np.array(
        [pwa_arr, systole_arr, diastole_arr, area_arr,
         rr_arr, diff_pwa_arr, diff_rr_arr],
        dtype=np.float32,
    )
