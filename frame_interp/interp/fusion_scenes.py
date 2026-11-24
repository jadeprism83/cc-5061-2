"""融合实验用的受控场景。

同一前景轮廓（带缺口的 C 形）分别扫过单色背景与细纹背景；另有交叉前景、
显露背景两类场景。所有场景在时间网格上直接渲染真值帧与真值前景掩码，
不经过任何插值。
"""

from __future__ import annotations

import numpy as np

BG = (0.12, 0.18, 0.28)
FG_YELLOW = (0.96, 0.82, 0.12)
FG_MAGENTA = (0.92, 0.12, 0.62)
FG_CYAN = (0.08, 0.86, 0.80)
FG_ORANGE = (0.95, 0.35, 0.08)


def _coords(h: int, w: int):
    ys, xs = np.mgrid[0:h, 0:w]
    return xs.astype(np.float64), ys.astype(np.float64)


def silhouette_mask(
    h: int, w: int, cx: float, cy: float, r_out: float = 24.0,
    r_in: float = 12.0, gap: float = 8.0,
) -> np.ndarray:
    """带右侧开口与顶部缺口的 C 形轮廓（非对称，利于暴露重影）。"""
    xs, ys = _coords(h, w)
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    ring = (d2 <= r_out ** 2) & (d2 >= r_in ** 2)
    opening = (xs > cx + 1.0) & (np.abs(ys - cy) < gap)
    notch = (np.abs(xs - cx) < 5.0) & (ys > cy - r_out) & (ys < cy - r_out + 9.0)
    return ring & ~opening & ~notch


def _fine_texture_bg(h: int, w: int, seed: int) -> np.ndarray:
    """细纹背景：低频唯一斑块（保证金字塔粗层可匹配）+ 2px 棋盘细纹 + 噪声。

    只有高频细纹的背景在 2x 降采样后会被平均成近似常数，块匹配在最粗尺度
    出现孔径歧义并污染整场运动；因此必须保留一个跨尺度可辨识的低频结构。
    """
    rng = np.random.default_rng(seed)
    xs, ys = _coords(h, w)
    checker = (((xs // 2).astype(int) + (ys // 2).astype(int)) % 2) * 2.0 - 1.0

    coarse_h, coarse_w = max(h // 8, 1), max(w // 8, 1)
    coarse = rng.normal(0.0, 1.0, (coarse_h, coarse_w))
    up = np.kron(coarse, np.ones((8, 8)))[:h, :w]
    for _ in range(4):
        up = (
            np.roll(up, 1, 0) + np.roll(up, -1, 0)
            + np.roll(up, 1, 1) + np.roll(up, -1, 1) + up
        ) / 5.0
    up = (up - up.mean()) / (up.std() + 1e-9)

    img = np.zeros((h, w, 3), dtype=np.float64)
    tint = np.array([0.9, 0.75, 1.15])
    for c in range(3):
        img[..., c] = (
            BG[c]
            + 0.105 * up * tint[c]
            + 0.032 * checker
            + rng.normal(0.0, 0.018, (h, w))
        )
    # 稀疏高对比锚点（3x3），进一步消除重复纹理歧义
    for _ in range(9):
        y = int(rng.integers(6, h - 9))
        x = int(rng.integers(6, w - 9))
        img[y:y + 3, x:x + 3] = np.clip(np.array(BG) * 2.4, 0, 1)
    return np.clip(img, 0.0, 1.0)


def _background(h: int, w: int, kind: str, seed: int) -> np.ndarray:
    if kind == "solid":
        img = np.zeros((h, w, 3), dtype=np.float64)
        img[:] = BG
        return img
    if kind == "texture":
        return _fine_texture_bg(h, w, seed)
    raise ValueError(f"未知背景类型 {kind}")


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def _sweep(
    bg_kind: str, h: int, w: int, steps: int, dx: float, seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    bg = _background(h, w, bg_kind, seed)
    frames, masks = [], []
    positions = []
    cy = h * 0.5
    for k in range(steps):
        u = k / (steps - 1)
        cx = 54.0 + dx * u
        m = silhouette_mask(h, w, cx, cy)
        img = bg.copy()
        img[m] = FG_YELLOW
        frames.append(_to_uint8(img))
        masks.append(m)
        positions.append([round(float(cx), 2), round(float(cy), 2)])
    meta = {
        "name": f"sweep_{bg_kind}",
        "family": "sweep",
        "background": bg_kind,
        "fg_colors": [FG_YELLOW],
        "steps": steps,
        "positions": positions,
        "note": "同一 C 形轮廓匀速横扫；前景为纯色，便于按颜色判定覆盖。",
    }
    return frames, masks, meta


def sweep_solid(h=144, w=192, steps=5, dx=26.0, seed=11):
    return _sweep("solid", h, w, steps, dx, seed)


def sweep_texture(h=144, w=192, steps=5, dx=26.0, seed=12):
    return _sweep("texture", h, w, steps, dx, seed)


def _crossing(
    bg_kind: str, h: int, w: int, steps: int, seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    bg = _background(h, w, bg_kind, seed)
    frames, masks = [], []
    positions = []
    cy = h * 0.5
    for k in range(steps):
        u = k / (steps - 1)
        front_x = 44.0 + 56.0 * u
        back_x = 148.0 - 56.0 * u
        back = silhouette_mask(h, w, back_x, cy)
        front = silhouette_mask(h, w, front_x, cy)
        img = bg.copy()
        img[back] = FG_CYAN
        img[front] = FG_MAGENTA
        frames.append(_to_uint8(img))
        masks.append(front | back)
        positions.append({
            "front": [round(float(front_x), 2), float(cy)],
            "back": [round(float(back_x), 2), float(cy)],
        })
    meta = {
        "name": f"crossing_{bg_kind}",
        "family": "crossing",
        "background": bg_kind,
        "fg_colors": [FG_MAGENTA, FG_CYAN],
        "steps": steps,
        "positions": positions,
        "note": "洋红轮廓始终在前，青色轮廓在后；中点完全重叠，是本质深度歧义。",
    }
    return frames, masks, meta


def crossing_solid(h=144, w=192, steps=5, seed=13):
    return _crossing("solid", h, w, steps, seed)


def crossing_texture(h=144, w=192, steps=5, seed=14):
    return _crossing("texture", h, w, steps, seed)


def _reveal(
    bg_kind: str, h: int, w: int, steps: int, seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    bg = _background(h, w, bg_kind, seed)
    frames, masks = [], []
    positions = []
    cy = h * 0.5
    half = 26
    for k in range(steps):
        u = k / (steps - 1)
        cx = 78.0 + 32.0 * u
        m = np.zeros((h, w), dtype=bool)
        x0, x1 = int(round(cx - half)), int(round(cx + half))
        y0, y1 = int(round(cy - half)), int(round(cy + half))
        m[y0:y1, x0:x1] = True
        img = bg.copy()
        img[m] = FG_ORANGE
        frames.append(_to_uint8(img))
        masks.append(m)
        positions.append([round(float(cx), 2), round(float(cy), 2)])
    meta = {
        "name": f"reveal_{bg_kind}",
        "family": "reveal",
        "background": bg_kind,
        "fg_colors": [FG_ORANGE],
        "steps": steps,
        "positions": positions,
        "note": "实心方块右移，左侧背景被显露；该区域在 I0 被遮挡、在 I1 可见。",
    }
    return frames, masks, meta


def reveal_solid(h=144, w=192, steps=5, seed=15):
    return _reveal("solid", h, w, steps, seed)


def reveal_texture(h=144, w=192, steps=5, seed=16):
    return _reveal("texture", h, w, steps, seed)


def all_fusion_scenes(h=144, w=192, steps=5):
    return [
        sweep_solid(h, w, steps),
        sweep_texture(h, w, steps),
        crossing_solid(h, w, steps),
        crossing_texture(h, w, steps),
        reveal_solid(h, w, steps),
        reveal_texture(h, w, steps),
    ]
