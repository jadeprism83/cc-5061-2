"""融合原型的分层可视化：每层都来自 FusionOutput 的显式掩码。"""

from __future__ import annotations

import numpy as np

from .core import to_float_rgb, to_uint8


def _base(out) -> np.ndarray:
    return to_float_rgb(out.frame)


def overlay(out) -> np.ndarray:
    """最终帧叠加：紫=两侧冲突，红=补全，橙=低可信切换边界。"""
    img = _base(out).copy()
    conflict = out.conflict & out.hit
    img[conflict] = img[conflict] * 0.45 + np.array([0.55, 0.25, 0.75]) * 0.55
    img[out.fill] = img[out.fill] * 0.35 + np.array([0.90, 0.10, 0.10]) * 0.65
    low = (~out.conflict) & (~out.fill) & (out.confidence < 0.4)
    img[low] = img[low] * 0.55 + np.array([0.95, 0.55, 0.10]) * 0.45
    return to_uint8(img)


def source_side_map(out) -> np.ndarray:
    """选源侧：绿=侧0，蓝=侧1，黄=两侧一致，红=补全，灰=空。"""
    h, w = out.side_choice.shape
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[:] = [0.12, 0.12, 0.14]
    img[out.side_choice == 0] = [0.10, 0.80, 0.20]
    img[out.side_choice == 1] = [0.10, 0.60, 0.90]
    img[out.side_choice == 3] = [0.90, 0.90, 0.20]
    img[out.fill] = [0.90, 0.15, 0.10]
    return to_uint8(img)


def conflict_map(out) -> np.ndarray:
    """冲突/可见性证据：白=正常，紫=两侧冲突，红=补全。"""
    h, w = out.conflict.shape
    img = np.full((h, w, 3), 0.95, dtype=np.float64)
    img[out.fill] = [0.90, 0.15, 0.10]
    img[out.conflict] = [0.55, 0.25, 0.75]
    return to_uint8(img)


def confidence_map(out) -> np.ndarray:
    c = np.clip(out.confidence, 0.0, 1.0)
    h, w = c.shape
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[..., 1] = c
    img[..., 0] = 0.15 * (1.0 - c)
    return to_uint8(img)


def all_layers(out) -> dict[str, np.ndarray]:
    return {
        "frame": out.frame,
        "overlay": overlay(out),
        "source_side": source_side_map(out),
        "conflict": conflict_map(out),
        "confidence": confidence_map(out),
    }
