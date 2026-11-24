"""与已知中间真值对比的量化度量。

- 像素误差：MAE（0-255）、PSNR。
- 边缘重影：在真值边缘带内，度量插值结果的梯度方向紊乱与双侧残影强度。
- 空洞范围：空洞/可回采/待填占比，以及补全区相对真值的误差。
"""

from __future__ import annotations

import numpy as np

from .core import InterpolationResult, SampleSource, to_float_rgb


def _luma(img: np.ndarray) -> np.ndarray:
    a = to_float_rgb(img)
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def pixel_errors(pred: np.ndarray, gt: np.ndarray) -> dict:
    p = to_float_rgb(pred)
    g = to_float_rgb(gt)
    mae = float(np.mean(np.abs(p - g)) * 255.0)
    mse = float(np.mean((p - g) ** 2))
    psnr = float("inf") if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)
    # 每通道最大偏差
    max_err = float(np.max(np.abs(p - g)) * 255.0)
    return {"mae_0_255": mae, "psnr_db": psnr, "mse": mse, "max_err": max_err}


def _edge_band(mask: np.ndarray, dilate: int = 2) -> np.ndarray:
    """二值掩码的边缘带（外扩 dilate 像素）。"""
    h, w = mask.shape
    edge = np.zeros_like(mask)
    m = mask.astype(np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(m, dy, 0), dx, 1)
            edge |= (shifted != m)
    band = edge.copy()
    for _ in range(dilate):
        grow = band.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            grow |= np.roll(np.roll(band, dy, 0), dx, 1)
        band = grow
    return band.astype(bool)


def ghosting_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """边缘重影度量。

    思路：真值中运动物体的边缘带是重影最明显的位置。
      1. edge_mae：边缘带内的平均像素误差（越大说明残影/双边越严重）；
      2. gradient_mismatch：预测与真值在边缘带内 Sobel 梯度方向的平均夹角差异；
      3. double_edge：预测边缘带内出现“次峰”的比例（局部极值点两侧都有强梯度）。
    """
    pl = _luma(pred)
    gl = _luma(gt)

    # 用亮度强响应位置近似运动物体边缘
    gx = np.zeros_like(gl); gy = np.zeros_like(gl)
    gx[:, 1:-1] = gl[:, 2:] - gl[:, :-2]
    gy[1:-1, :] = gl[2:, :] - gl[:-2, :]
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    edge = gmag > np.percentile(gmag, 92)
    band = _edge_band(edge, dilate=2)

    edge_mae = float(np.mean(np.abs(pl - gl)[band]) * 255.0)

    px = np.zeros_like(pl); py = np.zeros_like(pl)
    px[:, 1:-1] = pl[:, 2:] - pl[:, :-2]
    py[1:-1, :] = pl[2:, :] - pl[:-2, :]
    dot = gx * px + gy * py
    denom = np.sqrt((gx ** 2 + gy ** 2) * (px ** 2 + py ** 2)) + 1e-9
    cosang = np.clip(dot / denom, -1.0, 1.0)
    grad_mismatch = float(np.mean(1.0 - cosang[band]))

    # 双边检测：水平/垂直方向上，强梯度点两侧 2 像素内仍有强梯度
    pmag = np.sqrt(px ** 2 + py ** 2)
    thr = np.percentile(pmag[band], 70)
    strong = pmag > thr
    double_h = strong & np.roll(strong, 2, axis=1) & np.roll(strong, -2, axis=1)
    double_v = strong & np.roll(strong, 2, axis=0) & np.roll(strong, -2, axis=0)
    double_edge = float(np.mean((double_h | double_v)[band]))

    return {
        "edge_mae_0_255": edge_mae,
        "gradient_mismatch": grad_mismatch,
        "double_edge_ratio": double_edge,
        "edge_band_px": int(band.sum()),
    }


def hole_metrics(result: InterpolationResult, gt: np.ndarray) -> dict:
    """空洞范围与补全质量。"""
    h, w = result.sample_source.shape
    n = h * w
    inpaint = result.sample_source == SampleSource.INPAINT.value
    recoverable = result.holes.recoverable
    hole_all = ~result.grid.hit
    out = {
        "hole_ratio": float(np.mean(hole_all)),
        "recoverable_ratio": float(np.mean(recoverable)),
        "unrecoverable_inpaint_ratio": float(np.mean(inpaint)),
        "collision_ratio": float(np.mean(result.grid.collision)),
        "covered_ratio": float(np.mean(result.grid.covered)),
        "mean_confidence": float(np.mean(result.confidence)),
        "min_inpaint_confidence": float(
            result.confidence[inpaint].min() if np.any(inpaint) else 1.0
        ),
        "max_inpaint_confidence": float(
            result.confidence[inpaint].max() if np.any(inpaint) else 1.0
        ),
    }
    if np.any(inpaint):
        err = np.abs(to_float_rgb(result.frame) - to_float_rgb(gt))
        out["inpaint_region_mae_0_255"] = float(np.mean(err[inpaint]) * 255.0)
    else:
        out["inpaint_region_mae_0_255"] = 0.0
    return out


def evaluate(result: InterpolationResult, gt: np.ndarray) -> dict:
    return {
        "pixel": pixel_errors(result.frame, gt),
        "ghosting": ghosting_metrics(result.frame, gt),
        "holes": hole_metrics(result, gt),
    }
