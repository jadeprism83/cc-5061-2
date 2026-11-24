"""分层可视化：运动场、样本有效性、遮挡/碰撞、空洞分类、可信图、最终帧。"""

from __future__ import annotations

import numpy as np

from .core import InterpolationResult, SampleSource, to_float_rgb, to_uint8


def _color_wheel(n: int = 55) -> np.ndarray:
    """Middlebury 风格光流色轮。"""
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    wheel = np.zeros((n, 3), dtype=np.float64)
    col = 0
    for i in range(ry):
        wheel[col] = [255, 255 * i / ry, 0]; col += 1
    for i in range(yg):
        wheel[col] = [255 * (1 - i / yg), 255, 0]; col += 1
    for i in range(gc):
        wheel[col] = [0, 255, 255 * i / gc]; col += 1
    for i in range(cb):
        wheel[col] = [0, 255 * (1 - i / cb), 255]; col += 1
    for i in range(bm):
        wheel[col] = [255 * i / bm, 0, 255]; col += 1
    for i in range(mr):
        wheel[col] = [255, 0, 255 * (1 - i / mr)]; col += 1
    return wheel


_WHEEL = _color_wheel()


def flow_to_color(flow: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """把 (H,W,2) 位移场编码为 uint8 RGB。"""
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    if max_mag is None:
        max_mag = max(float(np.percentile(mag, 99)), 1e-6)
    ang = np.arctan2(-fy, -fx) / np.pi
    fk = (ang + 1.0) / 2.0 * (len(_WHEEL) - 1)
    k0 = np.floor(fk).astype(int) % len(_WHEEL)
    k1 = (k0 + 1) % len(_WHEEL)
    frac = fk - np.floor(fk)
    c0 = _WHEEL[k0]
    c1 = _WHEEL[k1]
    rgb = c0 * (1 - frac)[..., None] + c1 * frac[..., None]
    rad = np.minimum(mag / max_mag, 1.0)
    # 远离中心逐渐降低亮度，静止为白色
    rgb = 1.0 - rad[..., None] * (1.0 - rgb / 255.0)
    rgb = np.where(mag[..., None] <= 1e-9, 1.0, rgb)
    return to_uint8(rgb)


def validity_map(result: InterpolationResult) -> np.ndarray:
    """样本有效性分层：
    绿=单侧有效命中，蓝=两侧一致命中，黄=外推回采，红=有限补全（不确定），
    黑=无样本（理论上补全后不存在）。"""
    h, w = result.sample_source.shape
    out = np.zeros((h, w, 3), dtype=np.float64)
    ss = result.sample_source
    out[ss == SampleSource.SIDE0.value] = [0.1, 0.8, 0.2]
    out[ss == SampleSource.SIDE1.value] = [0.1, 0.6, 0.9]
    out[ss == SampleSource.BOTH.value] = [0.9, 0.9, 0.2]
    out[ss == SampleSource.INPAINT.value] = [0.9, 0.15, 0.1]
    return to_uint8(out)


def occlusion_map(result: InterpolationResult) -> np.ndarray:
    """碰撞/遮挡关系：白=无碰撞，橙=碰撞，紫=前景覆盖背景，灰=空洞。"""
    g = result.grid
    h, w = g.hit.shape
    out = np.full((h, w, 3), 0.95, dtype=np.float64)
    out[g.collision] = [1.0, 0.55, 0.1]
    out[g.covered & g.hit] = [0.55, 0.25, 0.75]
    out[~g.hit] = [0.35, 0.35, 0.35]
    return to_uint8(out)


def holes_map(result: InterpolationResult) -> np.ndarray:
    """空洞分类：黄=可回采，红=待填（有限补全），深绿=正常命中。"""
    h, w = result.grid.hit.shape
    out = np.zeros((h, w, 3), dtype=np.float64)
    out[:] = [0.05, 0.25, 0.1]
    out[result.holes.recoverable] = [0.95, 0.8, 0.1]
    out[result.holes.unrecoverable] = [0.85, 0.1, 0.1]
    return to_uint8(out)


def confidence_map(result: InterpolationResult) -> np.ndarray:
    """可信图：0=黑（不确定/补全），1=亮绿（真实观测）。"""
    c = np.clip(result.confidence, 0.0, 1.0)
    h, w = c.shape
    out = np.zeros((h, w, 3), dtype=np.float64)
    out[..., 1] = c
    out[..., 0] = 0.15 * (1.0 - c)
    return to_uint8(out)


def overlay_holes_on_frame(result: InterpolationResult) -> np.ndarray:
    """最终帧上叠加遮挡/补全区：红=待填补全，黄=可回采，紫=覆盖碰撞。"""
    base = to_float_rgb(result.frame)
    overlay = base.copy()
    overlay[result.holes.unrecoverable] = (
        overlay[result.holes.unrecoverable] * 0.35 + np.array([0.9, 0.1, 0.1]) * 0.65
    )
    overlay[result.holes.recoverable] = (
        overlay[result.holes.recoverable] * 0.45 + np.array([0.95, 0.8, 0.1]) * 0.55
    )
    cover = result.grid.covered & result.grid.hit & ~result.holes.unrecoverable
    overlay[cover] = overlay[cover] * 0.55 + np.array([0.55, 0.25, 0.75]) * 0.45
    return to_uint8(overlay)


def all_layers(result: InterpolationResult) -> dict[str, np.ndarray]:
    return {
        "frame": result.frame,
        "frame_with_occlusion": overlay_holes_on_frame(result),
        "motion_01": flow_to_color(result.flow.flow01),
        "motion_10": flow_to_color(result.flow.flow10),
        "validity": validity_map(result),
        "occlusion": occlusion_map(result),
        "holes": holes_map(result),
        "confidence": confidence_map(result),
    }
