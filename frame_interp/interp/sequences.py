"""带已知中间真值的合成测试序列。

每个序列返回 (frames, meta)：
  frames: list[np.ndarray uint8]，长度由 steps 决定，frames[0]=I0, frames[-1]=I1，
          中间帧即 ground-truth（同一渲染器在对应时刻直接绘制，非插值得到）。
  meta: 序列说明（物体轨迹、尺寸等）。
场景：
  1. moving_shapes：纯色背景上平移的圆与方块（含露出背景）。
  2. crossing：两个物体相向运动并交叉（遮挡关系交换）。
  3. exit_frame：物体运动出画，留下不可观测/可回采的背景区域。
  4. scene_cut：两幅完全无关画面，用于验证拒绝插值。
"""

from __future__ import annotations

import numpy as np

BG = (0.12, 0.18, 0.28)
RED = (0.9, 0.2, 0.15)
CYAN = (0.15, 0.8, 0.85)
YELLOW = (0.95, 0.8, 0.2)
MAGENTA = (0.8, 0.25, 0.7)
GREEN = (0.3, 0.75, 0.3)


def _blank(h: int, w: int, color=BG) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[:] = color
    return img


def _draw_disc(img, cx, cy, r, color):
    h, w = img.shape[:2]
    ys, xs = np.ogrid[:h, :w]
    m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
    img[m] = color


def _draw_rect(img, cx, cy, rw, rh, color):
    h, w = img.shape[:2]
    x0 = int(round(cx - rw)); x1 = int(round(cx + rw))
    y0 = int(round(cy - rh)); y1 = int(round(cy + rh))
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = color


def _textured_bg(h, w, seed=0):
    """带非周期纹理的静态背景，避免块匹配在重复条纹上产生歧义。"""
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3))
    base[:] = BG
    # 平滑随机斑块（多次盒式滤波）+ 细粒度噪声，整体非周期
    coarse = rng.normal(0, 1, (max(h // 8, 1), max(w // 8, 1)))
    up = np.kron(coarse, np.ones((8, 8)))[:h, :w]
    for _ in range(4):
        up = (
            np.roll(up, 1, 0) + np.roll(up, -1, 0)
            + np.roll(up, 1, 1) + np.roll(up, -1, 1) + up
        ) / 5.0
    up = (up - up.mean()) / (up.std() + 1e-9)
    shade = 0.12 * up
    for c in range(3):
        base[..., c] = np.clip(BG[c] + shade + rng.normal(0, 0.015, (h, w)), 0, 1)
    # 少量静态标志块，提供稀疏但唯一的锚点
    for i in range(6):
        y = int(rng.integers(10, h - 10)); x = int(rng.integers(10, w - 10))
        base[y:y + 3, x:x + 3] = np.clip(np.array(BG) * 1.9, 0, 1)
    return base


def moving_shapes(h=144, w=192, steps=5, seed=1):
    """圆从左向右平移，方块反方向小幅下移；背景为静态纹理。"""
    bg = _textured_bg(h, w, seed)
    frames = []
    positions = []
    for k in range(steps):
        t = k / (steps - 1)
        img = bg.copy()
        cx = 30 + t * 72
        cy = h * 0.38
        rx = 150 - t * 72
        ry = h * 0.72 + t * 18
        _draw_disc(img, cx, cy, 16, RED)
        _draw_rect(img, rx, ry, 12, 12, CYAN)
        frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
        positions.append({"disc": (cx, cy), "rect": (rx, ry), "t": t})
    return frames, {"name": "moving_shapes", "positions": positions}


def crossing(h=144, w=192, steps=5, seed=2):
    """红盘右行、青盘左行，在画面中部交叉；红盘始终在前（遮挡青盘）。"""
    bg = _textured_bg(h, w, seed)
    frames = []
    positions = []
    for k in range(steps):
        t = k / (steps - 1)
        img = bg.copy()
        c1x = 36 + t * 72
        c2x = 108 - t * 72
        cy = h // 2
        _draw_disc(img, c2x, cy, 18, CYAN)   # 后景先画
        _draw_disc(img, c1x, cy, 18, RED)    # 前景覆盖
        frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
        positions.append({"front": (c1x, cy), "back": (c2x, cy), "t": t})
    return frames, {"name": "crossing", "positions": positions,
                    "note": "红盘为前景，交叉时覆盖青盘"}


def exit_frame(h=144, w=192, steps=5, seed=3):
    """大方块从画面中部向右运动出画，露出其后方从未在 I0 观测到的背景。"""
    bg = _textured_bg(h, w, seed)
    frames = []
    positions = []
    for k in range(steps):
        t = k / (steps - 1)
        img = bg.copy()
        cx = 60 + t * 72   # 末端大部分出画
        cy = h // 2
        _draw_rect(img, cx, cy, 26, 26, YELLOW)
        frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
        positions.append({"rect": (cx, cy), "t": t})
    return frames, {"name": "exit_frame", "positions": positions,
                    "note": "方块出画，尾部背景在 I0 完全被遮挡"}


def scene_cut(h=144, w=192, seed=4):
    """两幅完全无关的画面：条纹场景 vs 噪声色块场景。"""
    rng = np.random.default_rng(seed)
    a = _textured_bg(h, w, seed)
    _draw_disc(a, 40, 40, 20, RED)
    b = np.zeros((h, w, 3))
    b[:h // 2] = MAGENTA
    b[h // 2:] = GREEN
    for _ in range(12):
        x, y = rng.integers(0, w), rng.integers(0, h)
        _draw_rect(b, x, y, 8, 8, YELLOW)
    return [
        (np.clip(a, 0, 1) * 255).astype(np.uint8),
        (np.clip(b, 0, 1) * 255).astype(np.uint8),
    ], {"name": "scene_cut", "note": "无关镜头，插值必须被拒绝"}


def all_sequences(h=144, w=192, steps=5):
    seqs = [
        moving_shapes(h, w, steps),
        crossing(h, w, steps),
        exit_frame(h, w, steps),
    ]
    cut = scene_cut(h, w)
    return seqs, cut
