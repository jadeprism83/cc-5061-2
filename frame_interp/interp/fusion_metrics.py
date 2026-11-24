"""融合实验指标：错误遮盖面积、重影面积、空洞率。

全部为逐像素计数，可由 tests/run_fusion_experiment.py 复算并回读 JSON 核对。
给定已知真值前景掩码时：
  - 错误遮盖：真值背景格上出现前景颜色（前景外像素被判为前景）。
  - 漏遮/欠遮盖：真值前景格上出现背景颜色。
  - 重影：真值背景格上，输出颜色既不属于背景统计也不属于任一前景色，
    或同一轮廓两侧同时残留前景色（双层）。
无真值掩码时退化为颜色域判定（不属于背景、也不属于前景调色板）。
"""

from __future__ import annotations

import numpy as np

from .core import to_float_rgb

FG_PALETTES = {
    "sweep_solid": [(0.96, 0.82, 0.12)],
    "sweep_texture": [(0.96, 0.82, 0.12)],
    "crossing_solid": [(0.92, 0.12, 0.62), (0.08, 0.86, 0.80)],
    "crossing_texture": [(0.92, 0.12, 0.62), (0.08, 0.86, 0.80)],
    "reveal_solid": [(0.95, 0.35, 0.08)],
    "reveal_texture": [(0.95, 0.35, 0.08)],
}


def _dist_to_palette(rgb: np.ndarray, palette) -> np.ndarray:
    d = np.full(rgb.shape[:2], np.inf)
    for c in palette:
        d = np.minimum(d, np.linalg.norm(rgb - np.asarray(c), axis=-1))
    return d


def foreground_evidence(frame: np.ndarray, palette) -> np.ndarray:
    """输出像素是否为前景色（纯色前景，容差 0.25）。"""
    rgb = to_float_rgb(frame)
    return _dist_to_palette(rgb, palette) < 0.25


def background_evidence(frame: np.ndarray, bg_color, tol: float = 0.20) -> np.ndarray:
    rgb = to_float_rgb(frame)
    return np.linalg.norm(rgb - np.asarray(bg_color), axis=-1) < tol


def occlusion_metrics(
    frame: np.ndarray,
    gt_mask: np.ndarray,
    gt_bg: np.ndarray,
    palette,
    bg_color,
) -> dict:
    """以真值前景掩码与真值背景为参照的遮盖错误。"""
    pred_fg = foreground_evidence(frame, palette)
    gt_fg = gt_mask.astype(bool)
    gt_bg_mask = gt_bg.astype(bool)
    n = frame.shape[0] * frame.shape[1]
    wrong_cover = pred_fg & gt_bg_mask
    under_cover = (~pred_fg) & gt_fg
    return {
        "wrong_cover_ratio": float(wrong_cover.sum() / n),
        "wrong_cover_px": int(wrong_cover.sum()),
        "under_cover_ratio": float(under_cover.sum() / n),
        "under_cover_px": int(under_cover.sum()),
        "gt_fg_ratio": float(gt_fg.mean()),
        "pred_fg_ratio": float(pred_fg.mean()),
    }


def ghosting_area(
    frame: np.ndarray,
    gt_mask: np.ndarray,
    gt_bg: np.ndarray,
    bg_stats: dict | None,
    palette,
    gt_frame: np.ndarray | None = None,
) -> dict:
    """重影面积：真值背景格上出现不属于该处真实背景的颜色。

    两条互补证据：
      1. 前景色残留：真值背景格出现前景调色板颜色（硬重影/错误遮盖）。
      2. 相对真值偏离：输出与“真值本帧同格颜色”之差超阈值（软重影）。
         细纹背景本身深浅起伏大，若拿全局均值当基准会把正常纹理误判为重影；
         因此用真值本帧颜色作基准，阈值取真值局部（7x7）标准差的 3 倍并设
         下限 0.10，只有明显偏离真实背景分布的像素才计入。
    """
    rgb = to_float_rgb(frame)
    gt_bg_mask = gt_bg.astype(bool)
    n = frame.shape[0] * frame.shape[1]
    fg_residue = foreground_evidence(frame, palette) & gt_bg_mask

    if gt_frame is not None:
        gt_rgb = to_float_rgb(gt_frame)
        local_std = _local_std(gt_rgb, radius=3)
        thr = np.maximum(3.0 * local_std, 0.10)
        dev = np.linalg.norm(rgb - gt_rgb, axis=-1)
        outlier = (dev > thr) & gt_bg_mask
    else:
        mu = np.asarray(bg_stats["mean"]) if bg_stats else np.zeros(3)
        sd = float(bg_stats["std"]) if bg_stats else 0.0
        thr = max(3.0 * sd, 0.10)
        dev = np.linalg.norm(rgb - mu, axis=-1)
        outlier = (dev > thr) & gt_bg_mask
    ghost = fg_residue | outlier
    return {
        "ghosting_ratio": float(ghost.mean()),
        "ghosting_px": int(ghost.sum()),
        "fg_residue_px": int(fg_residue.sum()),
        "bg_outlier_px": int(outlier.sum()),
        "ghosting_basis_px": int(gt_bg_mask.sum()),
        "total_px": int(n),
    }


def _local_std(rgb: np.ndarray, radius: int = 3) -> np.ndarray:
    """局部逐通道标准差（盒式，边界按最近值钳制），返回 HxW 平均通道标准差。"""
    h, w = rgb.shape[:2]
    k = 2 * radius + 1
    pad = np.pad(rgb, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    csum = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    csum = np.concatenate([np.zeros((1, csum.shape[1], 3)), csum], axis=0)
    csum = np.concatenate([np.zeros((csum.shape[0], 1, 3)), csum], axis=1)
    csum2 = np.cumsum(np.cumsum(pad ** 2, axis=0), axis=1)
    csum2 = np.concatenate([np.zeros((1, csum2.shape[1], 3)), csum2], axis=0)
    csum2 = np.concatenate([np.zeros((csum2.shape[0], 1, 3)), csum2], axis=1)

    def box(c):
        return (c[k:k + h, k:k + w] - c[:h, k:k + w]
                - c[k:k + h, :w] + c[:h, :w])

    area = k * k
    mean = box(csum) / area
    var = box(csum2) / area - mean ** 2
    var = np.maximum(var, 0.0)
    return np.sqrt(var).mean(axis=-1)


def hole_rate(out) -> dict:
    """空洞率：无真实源样本的格占比；另给补全后仍低可信的占比。"""
    h, w = out.hit.shape
    n = h * w
    return {
        "hole_ratio": float((~out.hit).sum() / n),
        "hole_px": int((~out.hit).sum()),
        "hit_px": int(out.hit.sum()),
        "fill_px": int(out.fill.sum()),
        "fill_ratio": float(out.fill.sum() / n),
        "conflict_px": int(out.conflict.sum()),
        "conflict_ratio": float(out.conflict.sum() / n),
        "uncertain_ratio": float(out.uncertain.sum() / n),
        "total_px": int(n),
    }


def color_stats(bg: np.ndarray) -> dict:
    rgb = to_float_rgb(bg).reshape(-1, 3)
    return {"mean": [float(v) for v in rgb.mean(axis=0)],
            "std": float(rgb.std(axis=0).mean())}
