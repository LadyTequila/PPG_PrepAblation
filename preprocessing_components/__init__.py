"""
preprocessing_components/
=========================

预处理流水线的 7 个可重组组件，对应消融对比表的 C1~C7。

C1 时窗裁剪              → trim.py
C2 Hypopnea 二次确认     → confirm_hypopnea.py（含 flow_severity 子模块）
C3 信号归一化            → normalize.py
C4 滤波                  → filter.py（含 ppg_extraction）
C5 标签粒度 + C6 延迟    → label.py
C7 滑窗                  → windowing.py

主入口（preprocess_modular.py）通过参数组合调用这些组件。
"""

__all__ = [
    "trim",
    "confirm_hypopnea",
    "flow_severity",
    "normalize",
    "filter",
    "label",
    "windowing",
]
