"""中间帧合成内核。

核心流程（所有步骤均在本模块显式实现，不调用整段补帧命令）：
  1. estimate_bidirectional_flow：块匹配 + 光流前后向一致性，得到 I0->I1 / I1->I0 运动场。
  2. detect_scene_cut：依据匹配残差与一致性比例判定场景切换，切换时拒绝插值。
  3. forward_warp：按时间比例 t 把两侧像素“正向投影”到目标时间位置，记录每个目标格的
     源坐标 (sx, sy)、源侧 0/1、碰撞、遮挡与深度序。
  4. 碰撞：多个源落到同一目标格时按深度序（遮挡判定得到的前后关系）保留前景样本；
     遮挡：前景覆盖背景处只采纳前景样本。
  5. 空洞：区分“可回采”（另一帧在 t 时刻能观测到，经外推重投影验证）与“待填”
     （两侧都未观测）；待填区仅使用有限次扩散补全并在 confidence 中标记不确定。
  6. 不做两原图的直接透明混合；每个输出像素都带可验证的源目标坐标与时间比例。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def to_float_rgb(frame: np.ndarray) -> np.ndarray:
    """转为 HxWx3 float64，范围 [0, 1]。"""
    a = np.asarray(frame)
    if a.dtype == np.uint8:
        a = a.astype(np.float64) / 255.0
    else:
        a = a.astype(np.float64)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    return np.clip(a, 0.0, 1.0)


def to_uint8(frame: np.ndarray) -> np.ndarray:
    return np.clip(np.round(frame * 255.0), 0, 255).astype(np.uint8)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """在浮点坐标 (x, y) 处双线性采样；越界返回 nan。"""
    h, w = img.shape[:2]
    out = np.full(x.shape + (img.shape[2],), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        x0 = np.where(np.isfinite(x), np.floor(x), 0.0).astype(np.int64)
        y0 = np.where(np.isfinite(y), np.floor(y), 0.0).astype(np.int64)
    valid = (
        (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
        & np.isfinite(x) & np.isfinite(y)
    )
    if not np.any(valid):
        return out
    xv, yv, xx, yy = x[valid], y[valid], x0[valid], y0[valid]
    wx, wy = xv - xx, yv - yy
    p00 = img[yy, xx]
    p10 = img[yy, xx + 1]
    p01 = img[yy + 1, xx]
    p11 = img[yy + 1, xx + 1]
    top = p00 * (1 - wx)[:, None] + p10 * wx[:, None]
    bot = p01 * (1 - wx)[:, None] + p11 * wx[:, None]
    out[valid] = top * (1 - wy)[:, None] + bot * wy[:, None]
    return out


def bilinear_sample_gray(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """单通道双线性采样，越界 nan。"""
    h, w = img.shape
    out = np.full(x.shape, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        x0 = np.where(np.isfinite(x), np.floor(x), 0.0).astype(np.int64)
        y0 = np.where(np.isfinite(y), np.floor(y), 0.0).astype(np.int64)
    valid = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1) & np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return out
    xv, yv, xx, yy = x[valid], y[valid], x0[valid], y0[valid]
    wx, wy = xv - xx, yv - yy
    out[valid] = (
        img[yy, xx] * (1 - wx) * (1 - wy)
        + img[yy, xx + 1] * wx * (1 - wy)
        + img[yy + 1, xx] * (1 - wx) * wy
        + img[yy + 1, xx + 1] * wx * wy
    )
    return out


# ---------------------------------------------------------------------------
# 块匹配双向光流
# ---------------------------------------------------------------------------


@dataclass
class FlowResult:
    flow01: np.ndarray          # I0 像素 -> I1 坐标的位移 (H,W,2)，通道顺序 (dx, dy)
    flow10: np.ndarray          # I1 像素 -> I0
    residual01: np.ndarray      # I0 经 flow01 回采 I1 的块匹配代价 (H,W)
    residual10: np.ndarray
    consistent: np.ndarray      # 前后向一致性掩码 (H,W)，按 I0 网格
    consistency_ratio: float
    mean_residual: float


def _block_match(
    src_gray: np.ndarray,
    dst_gray: np.ndarray,
    block: int,
    search: int,
    init_flow: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """以 src 网格每个像素为中心做块匹配，返回位移 (dx,dy) 与匹配代价。

    代价在 dst 上按块做 SAD；搜索窗整数位移，零位移（静止）在同代价时优先。
    逐行处理以控制内存；init_flow 给出粗尺度初始位移，搜索窗在其附近展开。
    """
    h, w = src_gray.shape
    pad = block // 2
    margin = pad + search * 3 + 4
    # src 只做 block/2 填充；dst 做 margin 填充以容纳初始大位移。
    # 两者窗口的“未填充坐标”索引统一通过 +pad / +margin 换算。
    src_pad = np.pad(src_gray, pad, mode="edge")
    dst_pad = np.pad(dst_gray, margin, mode="edge")
    Hp, Wp = dst_pad.shape
    flow = np.zeros((h, w, 2), dtype=np.float64)
    cost = np.zeros((h, w), dtype=np.float64)
    n = block * block
    shifts = [
        (dy, dx)
        for dy in range(-search, search + 1)
        for dx in range(-search, search + 1)
    ]
    left_all = np.arange(w)
    col_off = np.arange(block)
    base_dx = (
        np.rint(init_flow[..., 0]).astype(int) if init_flow is not None
        else np.zeros((h, w), dtype=int)
    )
    base_dy = (
        np.rint(init_flow[..., 1]).astype(int) if init_flow is not None
        else np.zeros((h, w), dtype=int)
    )

    for cy in range(h):
        # 块中心未填充坐标 (cy, cx)。src_pad 行 r 对应未填充行 r-pad，
        # 要覆盖未填充行 cy-pad..cy+pad，切片起点为 cy。列方向滑动窗口
        # 第 cx 个窗口的中心对齐未填充列 cx。
        r0 = min(cy, src_pad.shape[0] - block)
        src_row = src_pad[r0: r0 + block, :]
        all_win = np.lib.stride_tricks.sliding_window_view(
            src_row, block, axis=1
        )
        src_windows = all_win.transpose(1, 0, 2).reshape(w, n)
        cand = np.empty((len(shifts), w, n), dtype=np.float64)
        valid_shift = np.ones((len(shifts), w), dtype=bool)
        for si, (dy, dx) in enumerate(shifts):
            # 目标块中心未填充坐标 (cy+dy+base_dy, cx+dx+base_dx)；
            # 块左上角 = 中心 - pad；dst_pad 索引 = 未填充坐标 + margin。
            center_y = cy + dy + base_dy[cy, 0]
            center_x = left_all + dx + base_dx[cy]
            top = center_y + margin - pad
            left = center_x + margin - pad
            bad_row = (top < 0) or (top + block > Hp)
            bad_cols = (left < 0) | (left + block > Wp)
            top_c = min(max(top, 0), Hp - block)
            left_c = np.clip(left, 0, Wp - block)
            if bad_row:
                valid_shift[si, :] = False
            valid_shift[si, bad_cols] = False
            rows = (top_c + np.arange(block))[None, :, None]       # 1,block,1
            cols = (left_c[:, None] + col_off[None, :])[:, None, :]  # W,1,block
            patch = dst_pad[rows, cols]                              # W,block,block
            cand[si] = patch.reshape(w, n)
        sad = np.mean(
            np.abs(cand - src_windows[None, :, :]), axis=-1
        )  # shifts, W
        sad[~valid_shift] = np.inf
        # 全无效列回退零位移（已含在 shifts 中，且其通常有效）
        all_bad = ~np.isfinite(sad).any(axis=0)
        best = np.argmin(np.where(np.isfinite(sad), sad, 1e9), axis=0)
        for cx in range(w):
            if all_bad[cx]:
                flow[cy, cx] = (base_dx[cy, cx], base_dy[cy, cx])
                cost[cy, cx] = 1.0
                continue
            dy, dx = shifts[int(best[cx])]
            flow[cy, cx] = (dx + base_dx[cy, cx], dy + base_dy[cy, cx])
            cost[cy, cx] = sad[best[cx], cx]
    return flow, cost


def _pyr_down(img: np.ndarray) -> np.ndarray:
    """2x 降采样（2x2 均值）。"""
    h2, w2 = img.shape[0] // 2, img.shape[1] // 2
    crop = img[: h2 * 2, : w2 * 2]
    return crop.reshape(h2, 2, w2, 2).mean(axis=(1, 3))


def _upsample_flow(flow: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """最近邻放大流场到目标尺寸，位移乘 2。"""
    h, w = flow.shape[:2]
    th, tw = target_hw
    up = np.repeat(np.repeat(flow, 2, axis=0), 2, axis=1)[:th, :tw]
    up = up.copy()
    up[..., 0] *= 2.0
    up[..., 1] *= 2.0
    return up


def hierarchical_block_match(
    src_gray: np.ndarray,
    dst_gray: np.ndarray,
    block: int = 7,
    search: int = 6,
    levels: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """粗到细块匹配：最粗尺度捕获大位移，逐层放大 2 倍并细化。"""
    src_pyr = [src_gray]
    dst_pyr = [dst_gray]
    for _ in range(levels - 1):
        src_pyr.append(_pyr_down(src_pyr[-1]))
        dst_pyr.append(_pyr_down(dst_pyr[-1]))

    init = None
    flow_l = cost_l = None
    for level in range(levels - 1, -1, -1):
        s, d = src_pyr[level], dst_pyr[level]
        lh, lw = s.shape
        if level == levels - 1:
            b = max(5, block // (2 ** max(level - 1, 0)))
            flow_l, cost_l = _block_match(s, d, b, search * 3, None)
        else:
            b = block if level == 0 else max(5, block - 2 * level)
            if b % 2 == 0:
                b += 1
            init = _upsample_flow(flow_l, (lh, lw))
            flow_l, cost_l = _block_match(s, d, b, search, init)
    assert flow_l.shape[:2] == src_gray.shape[:2]
    return flow_l, cost_l


def _flow_to_coords(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = flow.shape[:2]
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return xs.astype(np.float64) + flow[..., 0], ys.astype(np.float64) + flow[..., 1]


def _median_filter_flow(flow: np.ndarray, size: int = 5) -> np.ndarray:
    """对位移场做二维中值滤波，抑制平坦区域的块匹配散点（标准光流后处理）。"""
    r = size // 2
    padded = np.pad(flow, ((r, r), (r, r), (0, 0)), mode="edge")
    h, w = flow.shape[:2]
    out = np.zeros_like(flow)
    for cy in range(h):
        win = padded[cy: cy + size, :, :]                       # size, W+2r, 2
        sw = np.lib.stride_tricks.sliding_window_view(
            win, size, axis=1
        )                                                      # size, W, 2, size
        sw = sw.transpose(1, 0, 3, 2).reshape(w, size * size, 2)
        out[cy] = np.median(sw, axis=1)
    return out


def _color_components(rgb: np.ndarray, thresh: float = 0.12) -> np.ndarray:
    """4-邻接颜色连通域标注：相邻像素颜色差小于 thresh 视为同一区域。

    用于把纯色物体（圆盘/方块）从背景中分割出来，得到区域标签。
    """
    h, w = rgb.shape[:2]
    labels = -np.ones((h, w), dtype=np.int64)
    next_label = 0
    for cy in range(h):
        for cx in range(w):
            if labels[cy, cx] >= 0:
                continue
            # 泛洪填充
            stack = [(cy, cx)]
            labels[cy, cx] = next_label
            base = rgb[cy, cx]
            while stack:
                y, x = stack.pop()
                for ay, ax in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + ay, x + ax
                    if (
                        0 <= ny < h and 0 <= nx < w and labels[ny, nx] < 0
                        and np.max(np.abs(rgb[ny, nx] - base)) < thresh
                    ):
                        labels[ny, nx] = next_label
                        stack.append((ny, nx))
            next_label += 1
    return labels


def _fill_flow_by_region(
    flow: np.ndarray, reliable: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """在每个颜色连通域内，用可靠像素的主导（中位）运动填充不可靠像素。"""
    out = flow.copy()
    for lab in np.unique(labels):
        region = labels == lab
        rel = region & reliable
        if not np.any(rel):
            continue
        dom = np.median(flow[rel], axis=0)
        out[region & ~reliable] = dom
    return out


def _propagate_flow(
    flow: np.ndarray, reliable: np.ndarray, max_dist: int = 24
) -> np.ndarray:
    """把可靠像素的运动传播到平坦/被遮挡内部（最近邻外推，限定距离）。

    块匹配在纯色物体内部因孔径问题得到散乱位移；物体边缘因跨背景对比是可靠的。
    用两趟倒角（chamfer）距离变换求每个不可靠格最近的可靠格，并拷贝其流。
    """
    h, w = flow.shape[:2]
    BIG = 1e6
    label = -np.ones((h, w), dtype=np.int64)
    label[reliable] = np.flatnonzero(reliable.ravel())
    dist = np.where(reliable, 0.0, BIG)

    def pass_2d(forward: bool):
        ys = range(h) if forward else range(h - 1, -1, -1)
        xs = range(w) if forward else range(w - 1, -1, -1)
        for cy in ys:
            for cx in xs:
                if reliable[cy, cx]:
                    continue
                for ay, ax, cost in (
                    (-1, 0, 1.0), (0, -1, 1.0),
                    (-1, -1, 1.414), (-1, 1, 1.414),
                ):
                    ny, nx = cy + ay, cx + ax
                    if 0 <= ny < h and 0 <= nx < w:
                        nd = dist[ny, nx] + cost
                        if nd < dist[cy, cx]:
                            dist[cy, cx] = nd
                            label[cy, cx] = label[ny, nx]

    pass_2d(True)
    # 反向趟用反向邻居
    for cy in range(h - 1, -1, -1):
        for cx in range(w - 1, -1, -1):
            if reliable[cy, cx]:
                continue
            for ay, ax, cost in (
                (1, 0, 1.0), (0, 1, 1.0),
                (1, 1, 1.414), (1, -1, 1.414),
            ):
                ny, nx = cy + ay, cx + ax
                if 0 <= ny < h and 0 <= nx < w:
                    nd = dist[ny, nx] + cost
                    if nd < dist[cy, cx]:
                        dist[cy, cx] = nd
                        label[cy, cx] = label[ny, nx]

    flat_flow = flow.reshape(-1, 2)
    out = flow.copy()
    fillable = (label >= 0) & (dist <= max_dist) & ~reliable
    out[fillable] = flat_flow[label[fillable]]
    return out


def estimate_bidirectional_flow(
    img0: np.ndarray,
    img1: np.ndarray,
    block: int = 7,
    search: int = 6,
    levels: int = 3,
    median_size: int = 5,
    propagate_dist: int = 24,
) -> FlowResult:
    """分层块匹配双向估计 + 中值滤波 + 一致性校验 + 物体内部流传播。"""
    g0 = _luma(to_float_rgb(img0))
    g1 = _luma(to_float_rgb(img1))
    f01, r01 = hierarchical_block_match(g0, g1, block, search, levels)
    f10, r10 = hierarchical_block_match(g1, g0, block, search, levels)
    f01 = _median_filter_flow(f01, median_size)
    f10 = _median_filter_flow(f10, median_size)

    h, w = g0.shape
    # 第一轮一致性：找出可靠像素（前后向闭合且在画面内、残差低）
    x01, y01 = _flow_to_coords(f01)
    in_bounds = (x01 >= 0) & (x01 < w) & (y01 >= 0) & (y01 < h)
    back_dx = bilinear_sample_gray(f10[..., 0], x01, y01)
    back_dy = bilinear_sample_gray(f10[..., 1], x01, y01)
    roundtrip = np.sqrt(
        (f01[..., 0] + back_dx) ** 2 + (f01[..., 1] + back_dy) ** 2
    )
    reliable0 = (
        in_bounds & np.isfinite(roundtrip) & (roundtrip < 1.5) & (r01 < 0.08)
    )

    # 反向一致性（在 I1 网格）
    x10, y10 = _flow_to_coords(f10)
    in_b1 = (x10 >= 0) & (x10 < w) & (y10 >= 0) & (y10 < h)
    fwd_dx = bilinear_sample_gray(f01[..., 0], x10, y10)
    fwd_dy = bilinear_sample_gray(f01[..., 1], x10, y10)
    rt1 = np.sqrt((f10[..., 0] + fwd_dx) ** 2 + (f10[..., 1] + fwd_dy) ** 2)
    reliable1 = in_b1 & np.isfinite(rt1) & (rt1 < 1.5) & (r10 < 0.08)

    # 把边缘可靠运动传播到平坦物体内部
    f01 = _propagate_flow(f01, reliable0, propagate_dist)
    f10 = _propagate_flow(f10, reliable1, propagate_dist)

    # 颜色连通域内用主导运动补齐仍为背景流的物体内部
    labels0 = _color_components(to_float_rgb(img0))
    labels1 = _color_components(to_float_rgb(img1))
    f01 = _fill_flow_by_region(f01, reliable0, labels0)
    f10 = _fill_flow_by_region(f10, reliable1, labels1)

    # 传播后重新计算一致性掩码，供 splat 置信度使用
    x01, y01 = _flow_to_coords(f01)
    in_bounds = (x01 >= 0) & (x01 < w) & (y01 >= 0) & (y01 < h)
    back_dx = bilinear_sample_gray(f10[..., 0], x01, y01)
    back_dy = bilinear_sample_gray(f10[..., 1], x01, y01)
    roundtrip = np.sqrt(
        (f01[..., 0] + back_dx) ** 2 + (f01[..., 1] + back_dy) ** 2
    )
    consistent = in_bounds & np.isfinite(roundtrip) & (roundtrip < 1.5)
    consistency_ratio = float(np.mean(consistent))
    mean_residual = float(np.mean(r01) + np.mean(r10)) / 2.0
    return FlowResult(
        flow01=f01,
        flow10=f10,
        residual01=r01,
        residual10=r10,
        consistent=consistent,
        consistency_ratio=consistency_ratio,
        mean_residual=mean_residual,
    )


# ---------------------------------------------------------------------------
# 场景切换检测
# ---------------------------------------------------------------------------


class SceneCutError(ValueError):
    """两帧属于不同镜头，拒绝无意义插值。"""


@dataclass
class SceneCutReport:
    is_cut: bool
    consistency_ratio: float
    mean_residual: float
    global_diff: float
    reason: str


def detect_scene_cut(
    img0: np.ndarray,
    img1: np.ndarray,
    flow: FlowResult,
    consistency_threshold: float = 0.35,
    residual_threshold: float = 0.18,
    diff_threshold: float = 0.30,
) -> SceneCutReport:
    """场景切换判定。

    三条证据任一强证据即判切换：
      - 前后向一致像素比例过低（运动场对不上）；
      - 块匹配平均残差过高（对上的位置也长得不像）；
      - 全局归一化像素差过大且一致性低（内容整体替换）。
    """
    a = to_float_rgb(img0)
    b = to_float_rgb(img1)
    global_diff = float(np.mean(np.abs(a - b)))
    reasons = []
    if flow.consistency_ratio < consistency_threshold:
        reasons.append(
            f"一致像素比例 {flow.consistency_ratio:.2f} < {consistency_threshold}"
        )
    if flow.mean_residual > residual_threshold:
        reasons.append(
            f"匹配残差 {flow.mean_residual:.3f} > {residual_threshold}"
        )
    if global_diff > diff_threshold and flow.consistency_ratio < 0.6:
        reasons.append(
            f"全局差异 {global_diff:.3f} 且一致性仅 {flow.consistency_ratio:.2f}"
        )
    is_cut = len(reasons) > 0
    return SceneCutReport(
        is_cut=is_cut,
        consistency_ratio=flow.consistency_ratio,
        mean_residual=flow.mean_residual,
        global_diff=global_diff,
        reason="；".join(reasons) if reasons else "运动场与外观均连续",
    )


# ---------------------------------------------------------------------------
# 正向投影（splat）与碰撞 / 遮挡
# ---------------------------------------------------------------------------


class SampleSource(Enum):
    NONE = 0
    SIDE0 = 1     # 来自 I0
    SIDE1 = 2
    BOTH = 3      # 两侧在同一目标格一致命中（仅用于统计，融合取深度序胜者）
    INPAINT = 4   # 两侧均未观测，有限补全


@dataclass
class SplatGrid:
    """目标时间 t 网格上的投影记录。"""
    color: np.ndarray            # (H,W,3) 命中颜色
    src_x: np.ndarray            # 命中样本在源帧的浮点 x
    src_y: np.ndarray
    src_side: np.ndarray         # 0 / 1
    depth: np.ndarray            # 深度序，越小越靠前
    hit: np.ndarray              # 是否被任一侧命中
    collision: np.ndarray        # 同一目标格出现 >1 个投影
    covered: np.ndarray          # 前景样本覆盖背景样本的格
    confidence: np.ndarray       # 样本可信度 [0,1]
    n_hits_0: np.ndarray
    n_hits_1: np.ndarray


def _splat_one_side(
    src_rgb: np.ndarray,
    flow_full: np.ndarray,
    t: float,
    side: int,
    consistent: np.ndarray,
    h: int,
    w: int,
) -> dict:
    """把一侧像素按时间比例投影到 t 时刻（最近邻落格 + 子像素距离记录）。"""
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    scale = t if side == 0 else (1.0 - t)
    tx = xs.astype(np.float64) + flow_full[..., 0] * scale
    ty = ys.astype(np.float64) + flow_full[..., 1] * scale
    ix = np.rint(tx).astype(np.int64)
    iy = np.rint(ty).astype(np.int64)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    conf = np.where(consistent, 1.0, 0.45)

    return {
        "sx": xs.astype(np.float64),
        "sy": ys.astype(np.float64),
        "tx": tx,
        "ty": ty,
        "ix": ix,
        "iy": iy,
        "inside": inside,
        "color": src_rgb,
        "side": side,
        "scale": scale,
        "conf": conf,
    }


def _dedup_one_side(s: dict, h: int, w: int,
                    fg: np.ndarray | None = None) -> dict:
    """单侧正向投影的 2x2 子像素 splat。

    每个源像素按双线性权重落到相邻 4 个目标格；当同一侧多个不同表面的源
    （如运动物体与露出的背景）落到同一格时，保留权重最大的源；权重相等时
    用前景评分（fg）保留更靠前的表面。绝不在单侧内部混合颜色——跨表面冲突
    交给 _merge_splats 按遮挡关系裁决。
    """
    idx = s["inside"]
    tx, ty = s["tx"][idx], s["ty"][idx]
    x0 = np.floor(tx).astype(np.int64)
    y0 = np.floor(ty).astype(np.int64)
    fx, fy = tx - x0, ty - y0
    src_ix = s["sx"][idx].astype(int)
    src_iy = s["sy"][idx].astype(int)
    colors = s["color"][src_iy, src_ix]
    conf = s["conf"][idx]
    sx, sy = s["sx"][idx], s["sy"][idx]
    if fg is not None:
        fgv = fg[src_iy, src_ix]
    else:
        fgv = np.zeros_like(conf)

    # 统计每个目标格被多少个“不同表面”的源投影（颜色不同即视为不同表面）。
    # 纯双向流对对称交叉的两个表面没有深度信息，这种多表面争夺是本质歧义，
    # 必须显式标记并在可信图压到低可信，而不是宣称恢复了真实遮挡关系。
    cell_n = np.zeros((h, w), dtype=np.int32)
    np.add.at(cell_n, (s["iy"][idx], s["ix"][idx]), 1)

    # 光栅序键：y*w + x，作为最终确定性层序规则（保留源帧中先出现的表面）。
    key = (src_iy * w + src_ix).astype(np.int64)

    # 收集全部 2x2 splat 记录，再做全局排序选优。必须全局而非顺序覆盖：
    # 否则后写入的低分会淘汰先写入的高分，使被淘汰者再无翻盘机会。
    corners = [
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ]
    rec_g = []; rec_w = []; rec_fg = []; rec_key = []
    rec_c0 = []; rec_c1 = []; rec_c2 = []; rec_conf = []; rec_sx = []; rec_sy = []
    for ox, oy, wgt in corners:
        gx, gy = x0 + ox, y0 + oy
        valid = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        gx, gy = gx[valid], gy[valid]
        wv = wgt[valid]
        keep = wv > 1e-12
        gx, gy, wv = gx[keep], gy[keep], wv[keep]
        rec_g.append(gy * w + gx)
        rec_w.append(wv)
        rec_fg.append(fgv[valid][keep])
        rec_key.append(key[valid][keep])
        c = colors[valid][keep]
        rec_c0.append(c[:, 0]); rec_c1.append(c[:, 1]); rec_c2.append(c[:, 2])
        rec_conf.append(conf[valid][keep])
        rec_sx.append(sx[valid][keep]); rec_sy.append(sy[valid][keep])

    gid = np.concatenate(rec_g)
    n = gid.shape[0]
    # 排序优先级（升序）：目标格 -> 权重降 -> 评分降 -> 光栅键升。
    # 用复合结构化键实现：权重/评分取负以实现降序。
    order = np.lexsort((
        np.concatenate(rec_key),
        -np.concatenate(rec_fg),
        -np.concatenate(rec_w),
        gid,
    ))
    gid_s = gid[order]
    first = np.ones(n, dtype=bool)
    first[1:] = gid_s[1:] != gid_s[:-1]
    gsel = gid_s[first]
    ii = gsel // w
    jj = gsel - ii * w

    def pick(parts):
        return np.concatenate(parts)[order][first]

    win_color = np.stack([pick(rec_c0), pick(rec_c1), pick(rec_c2)], axis=-1)
    win_sx = pick(rec_sx)
    win_sy = pick(rec_sy)
    win_conf = pick(rec_conf)

    hit = np.zeros((h, w), dtype=bool)
    hit[ii, jj] = True
    # 同一目标格被多个源争夺即记为同侧歧义格（无论最终谁胜出）。
    ambiguous = cell_n > 1
    return {
        "ii": ii, "jj": jj,
        "color": win_color,
        "sx": win_sx, "sy": win_sy,
        "conf": win_conf,
        "side": s["side"], "scale": s["scale"],
        "ambiguous": ambiguous,
    }


def _source_occlusion_mask(
    flow: np.ndarray, consistent: np.ndarray
) -> np.ndarray:
    """计算源帧上的遮挡掩码：前向投影中多个源像素争夺同一目标格时，
    落在碰撞格且自身一致性较差的源像素视为被遮挡（前景把它盖掉）。

    返回 True 表示该源像素在投影方向上被遮挡。
    """
    h, w = flow.shape[:2]
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    tx = xs + flow[..., 0]
    ty = ys + flow[..., 1]
    ix = np.rint(tx).astype(np.int64)
    iy = np.rint(ty).astype(np.int64)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    counts = np.zeros((h, w), dtype=np.int32)
    np.add.at(counts, (iy[inside], ix[inside]), 1)
    collides = np.zeros((h, w), dtype=bool)
    collides[inside] = counts[iy[inside], ix[inside]] > 1
    # 碰撞格中，一致性差的源判为被遮挡；一致流（可靠前景）不判遮挡
    return collides & ~consistent


def _foreground_score(
    rgb: np.ndarray, flow: np.ndarray, reliable: np.ndarray,
    other_rgb: np.ndarray | None = None,
    back_flow: np.ndarray | None = None,
) -> np.ndarray:
    """逐像素前景评分（越大越靠前）。

    纯双向流在两个对称表面重叠处不含深度信息，因此评分**只采用几何证据**，
    不使用颜色反差/边缘强度这类外观启发式——对称的两个物体即便颜色不同，其
    外观特征也可能差异很大，用它们猜深度会把噪声当成遮挡关系。几何项：
      1. 运动幅度：区分“真正移动到该位置的运动表面”与“恰好落在同格的静态背景”。
         归一化到 [0,1]，所有明显运动表面都饱和为 1，因此两个对称运动表面打平。
      2. 运动一致性（reliable）：前后向闭合的可靠样本加分。
      3. 跨帧可见持续性：沿反向流能在另一帧找到颜色匹配，说明该表面在两帧都
         存活，不会在中间被永久遮挡。
    两个对称运动表面因此得到完全相同的评分，最终顺序交给确定性的参考帧（I0）
    光栅层序规则，并在可信图标记低可信——而非宣称从像素恢复了真实隐藏深度。
    评分仅用于碰撞格的前景裁决，不改变非碰撞像素。
    """
    h, w = rgb.shape[:2]
    # 归一化运动幅度：8px 以上位移饱和为 1，静态背景为 0。
    motion = np.minimum(
        np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2) / 8.0, 1.0
    )
    score = (
        2.0 * motion
        + 1.0 * reliable.astype(np.float64)
    )

    if other_rgb is not None and back_flow is not None:
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        ox = xs + back_flow[..., 0]
        oy = ys + back_flow[..., 1]
        sampled = bilinear_sample(other_rgb, ox.astype(np.float64),
                                  oy.astype(np.float64))
        match = np.isfinite(sampled).all(axis=-1) & (
            np.linalg.norm(sampled - rgb, axis=-1) < 0.18
        )
        score = score + 0.5 * match.astype(np.float64)
    return score


def _merge_splats(s0: dict, s1: dict, h: int, w: int,
                  occ0: np.ndarray, occ1: np.ndarray,
                  fg0: np.ndarray, fg1: np.ndarray) -> SplatGrid:
    """合并两侧投影。

    判定规则：
      - 仅一侧命中 -> 普通有效样本；
      - 两侧命中且颜色接近（同一表面被两侧共同观测）-> 一致命中，取两侧均值，
        不算碰撞、不算遮挡；
      - 两侧命中但颜色冲突（不同表面争夺同一格）-> 真碰撞，按几何遮挡关系决定
        前景：源帧上未被遮挡的一侧胜出，败者标记覆盖；两侧遮挡状态相同时回退
        到置信度。绝不做透明平均。
    """
    d0 = _dedup_one_side(s0, h, w, fg0)
    d1 = _dedup_one_side(s1, h, w, fg1)

    # 同侧多表面争夺格（本质深度歧义）：合并到统一歧义掩码，稍后压低可信度。
    amb_any = d0["ambiguous"] | d1["ambiguous"]

    grid = SplatGrid(
        color=np.zeros((h, w, 3)),
        src_x=np.full((h, w), np.nan),
        src_y=np.full((h, w), np.nan),
        src_side=np.full((h, w), -1, dtype=np.int64),
        depth=np.full((h, w), np.inf),
        hit=np.zeros((h, w), dtype=bool),
        collision=np.zeros((h, w), dtype=bool),
        covered=np.zeros((h, w), dtype=bool),
        confidence=np.zeros((h, w)),
        n_hits_0=np.zeros((h, w), dtype=np.int32),
        n_hits_1=np.zeros((h, w), dtype=np.int32),
    )
    grid.n_hits_0[d0["ii"], d0["jj"]] = 1
    grid.n_hits_1[d1["ii"], d1["jj"]] = 1

    # 每个侧样本在其源帧上的遮挡状态与前景评分（按最佳源坐标查）
    def _sample(field, d):
        sy = np.clip(np.rint(d["sy"]).astype(int), 0, h - 1)
        sx = np.clip(np.rint(d["sx"]).astype(int), 0, w - 1)
        return field[sy, sx]
    d0 = dict(d0, src_occ=_sample(occ0, d0), fg=_sample(fg0, d0))
    d1 = dict(d1, src_occ=_sample(occ1, d1), fg=_sample(fg1, d1))

    def place(d, grid):
        ii, jj = d["ii"], d["jj"]
        grid.hit[ii, jj] = True
        grid.color[ii, jj] = d["color"]
        grid.src_x[ii, jj] = d["sx"]
        grid.src_y[ii, jj] = d["sy"]
        grid.src_side[ii, jj] = d["side"]
        grid.confidence[ii, jj] = d["conf"]
        grid.depth[ii, jj] = 1.0 - d["conf"]

    place(d0, grid)

    ii1, jj1 = d1["ii"], d1["jj"]
    occupied = grid.hit[ii1, jj1]
    color_diff = np.linalg.norm(d1["color"] - grid.color[ii1, jj1], axis=-1)
    agree = occupied & (color_diff < 0.18)
    clash = occupied & (color_diff >= 0.18)

    # 一致命中：同一表面，平均颜色，置信度取高
    ay, ax = ii1[agree], jj1[agree]
    grid.color[ay, ax] = 0.5 * (grid.color[ay, ax] + d1["color"][agree])
    grid.confidence[ay, ax] = np.maximum(grid.confidence[ay, ax], d1["conf"][agree])
    grid.depth[ay, ax] = 1.0 - grid.confidence[ay, ax]

    # 真碰撞：几何遮挡优先；遮挡状态相同时用前景评分裁决，仍并列再看置信度
    occ0_grid = np.zeros((h, w), dtype=bool)
    fg0_grid = np.zeros((h, w))
    conf0_grid = np.zeros((h, w))
    occ0_grid[d0["ii"], d0["jj"]] = d0["src_occ"]
    fg0_grid[d0["ii"], d0["jj"]] = d0["fg"]
    conf0_grid[d0["ii"], d0["jj"]] = d0["conf"]
    cy, cx = ii1[clash], jj1[clash]
    grid.collision[cy, cx] = True
    side0_occluded = occ0_grid[cy, cx]
    side1_occluded = d1["src_occ"][clash]
    # 1) 明确的几何遮挡不对称
    side1_wins = side0_occluded & ~side1_occluded
    side0_wins = side1_occluded & ~side0_occluded
    # 2) 遮挡状态相同 -> 前景评分（含跨帧可见持续性）
    tie = ~(side1_wins | side0_wins)
    fg_diff = d1["fg"][clash] - fg0_grid[cy, cx]
    fg_better = fg_diff > 1e-6
    fg_worse = fg_diff < -1e-6
    side1_wins = side1_wins | (tie & fg_better)
    # 3) 前景评分相同（含置信度差异）：双向流对两个对称表面无深度信息，
    #    属本质歧义，确定性保留侧 0 样本，并在可信图标记不确定而非宣称恢复隐藏细节
    full_tie = tie & ~fg_better & ~fg_worse
    # 仅当评分有明确差距时才让置信度翻盘，避免用噪声置信度决定深度
    conf_flip = tie & fg_worse & (
        d1["conf"][clash] > conf0_grid[cy, cx] + 0.3
    )
    side1_wins = side1_wins | conf_flip

    wy, wx = cy[side1_wins], cx[side1_wins]
    grid.color[wy, wx] = d1["color"][clash][side1_wins]
    grid.src_x[wy, wx] = d1["sx"][clash][side1_wins]
    grid.src_y[wy, wx] = d1["sy"][clash][side1_wins]
    grid.src_side[wy, wx] = 1
    grid.confidence[wy, wx] = d1["conf"][clash][side1_wins]
    grid.depth[wy, wx] = 1.0 - grid.confidence[wy, wx]
    grid.covered[cy, cx] = True
    # 碰撞覆盖区无法从双向流确证深度，统一压到 0.5 以下标记不确定
    grid.confidence[cy, cx] = np.minimum(grid.confidence[cy, cx], 0.5)
    ty, tx = cy[full_tie], cx[full_tie]
    grid.confidence[ty, tx] = np.minimum(grid.confidence[ty, tx], 0.35)

    # 侧 1 独占格
    free = ~occupied
    fy, fx = ii1[free], jj1[free]
    grid.hit[fy, fx] = True
    grid.color[fy, fx] = d1["color"][free]
    grid.src_x[fy, fx] = d1["sx"][free]
    grid.src_y[fy, fx] = d1["sy"][free]
    grid.src_side[fy, fx] = 1
    grid.confidence[fy, fx] = d1["conf"][free]
    grid.depth[fy, fx] = 1.0 - grid.confidence[fy, fx]

    # 同侧多表面争夺（如对称交叉的两个圆盘精确重合）：双向流无深度信息，
    # 层序仅由确定性规则给出，必须在可信图标记为低可信，不宣称恢复真实遮挡。
    ay, ax = np.where(amb_any & grid.hit)
    grid.collision[ay, ax] = True
    grid.covered[ay, ax] = True
    grid.confidence[ay, ax] = np.minimum(grid.confidence[ay, ax], 0.35)
    return grid


# ---------------------------------------------------------------------------
# 空洞分类：可回采 vs 待填
# ---------------------------------------------------------------------------


@dataclass
class HoleMap:
    recoverable: np.ndarray     # 可由对侧外推重投影回采
    unrecoverable: np.ndarray   # 两侧都没观察到 -> 有限补全
    recover_side: np.ndarray    # 回采来源侧 0/1，-1 表示无
    recover_x: np.ndarray
    recover_y: np.ndarray


def classify_holes(
    grid: SplatGrid,
    flow01: np.ndarray,
    flow10: np.ndarray,
    img0: np.ndarray,
    img1: np.ndarray,
    t: float,
) -> HoleMap:
    h, w = grid.hit.shape
    hole = ~grid.hit
    recoverable = np.zeros_like(hole)
    recover_side = np.full((h, w), -1, dtype=np.int64)
    rec_x = np.full((h, w), np.nan)
    rec_y = np.full((h, w), np.nan)

    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xf, yf = xs.astype(np.float64), ys.astype(np.float64)

    # 从目标格反向延长两侧轨迹，检验“另一帧在 t 时刻是否能观测到该位置”。
    # I0 侧轨迹：x_t = x0 + t*d01。空洞格沿 -t*d01 外推回 I0 取流再验证。
    # 做法：对每个空洞格，分别假设它属于侧 0 / 侧 1 的轨迹延长线，
    # 在源帧采样流，再投影回 t，落点必须回到本格（闭环）且源坐标在画面内。
    f01_dx = flow01[..., 0]
    f01_dy = flow01[..., 1]
    f10_dx = flow10[..., 0]
    f10_dy = flow10[..., 1]

    # 侧 0 回采：x0 = x_t - t * d01(x0)。用零位移初始化做两次不动点迭代。
    x0_est = xf.copy()
    y0_est = yf.copy()
    for _ in range(3):
        dx = bilinear_sample_gray(f01_dx, x0_est, y0_est)
        dy = bilinear_sample_gray(f01_dy, x0_est, y0_est)
        x0_est = xf - t * dx
        y0_est = yf - t * dy
    dx0 = bilinear_sample_gray(f01_dx, x0_est, y0_est)
    dy0 = bilinear_sample_gray(f01_dy, x0_est, y0_est)
    back_t_x0 = x0_est + t * dx0
    back_t_y0 = y0_est + t * dy0
    close0 = (
        np.isfinite(back_t_x0)
        & (np.abs(back_t_x0 - xf) < 1.5)
        & (np.abs(back_t_y0 - yf) < 1.5)
        & (x0_est >= 0) & (x0_est < w)
        & (y0_est >= 0) & (y0_est < h)
    )

    # 侧 1 回采：x1 = x_t + (1-t) * d10(x1)
    x1_est = xf.copy()
    y1_est = yf.copy()
    for _ in range(3):
        dx = bilinear_sample_gray(f10_dx, x1_est, y1_est)
        dy = bilinear_sample_gray(f10_dy, x1_est, y1_est)
        x1_est = xf + (1.0 - t) * dx
        y1_est = yf + (1.0 - t) * dy
    dx1 = bilinear_sample_gray(f10_dx, x1_est, y1_est)
    dy1 = bilinear_sample_gray(f10_dy, x1_est, y1_est)
    back_t_x1 = x1_est - (1.0 - t) * dx1
    back_t_y1 = y1_est - (1.0 - t) * dy1
    close1 = (
        np.isfinite(back_t_x1)
        & (np.abs(back_t_x1 - xf) < 1.5)
        & (np.abs(back_t_y1 - yf) < 1.5)
        & (x1_est >= 0) & (x1_est < w)
        & (y1_est >= 0) & (y1_est < h)
    )

    # 静止背景快速回采：空洞格在两侧采样到的流都接近 0，说明该位置在两帧
    # 都是静态背景（仅正向投影漏采），任一侧同坐标即可直接回采。
    static0 = (
        np.isfinite(dx0) & np.isfinite(dy0)
        & (np.abs(dx0) < 1.0) & (np.abs(dy0) < 1.0)
    )
    static1 = (
        np.isfinite(dx1) & np.isfinite(dy1)
        & (np.abs(dx1) < 1.0) & (np.abs(dy1) < 1.0)
    )
    static_bg = hole & static0 & static1
    close0 = close0 | static_bg
    close1 = close1 | static_bg

    cand0 = hole & close0
    cand1 = hole & close1
    # 两侧都可回采时选闭环误差更小的一侧
    err0 = np.abs(back_t_x0 - xf) + np.abs(back_t_y0 - yf)
    err1 = np.abs(back_t_x1 - xf) + np.abs(back_t_y1 - yf)
    use0 = cand0 & (~cand1 | (err0 <= err1))
    use1 = cand1 & (~use0)

    recoverable[use0 | use1] = True
    recover_side[use0] = 0
    recover_side[use1] = 1
    rec_x[use0] = x0_est[use0]
    rec_y[use0] = y0_est[use0]
    rec_x[use1] = x1_est[use1]
    rec_y[use1] = y1_est[use1]

    unrecoverable = hole & ~recoverable
    return HoleMap(recoverable, unrecoverable, recover_side, rec_x, rec_y)


# ---------------------------------------------------------------------------
# 有限补全：限定迭代次数的邻域扩散，不宣称恢复隐藏真实细节
# ---------------------------------------------------------------------------


def limited_diffusion_inpaint(
    color: np.ndarray,
    known: np.ndarray,
    confidence: np.ndarray,
    max_iters: int = 60,
) -> tuple[np.ndarray, np.ndarray, int]:
    """对 unknown 区域做有限次 Laplace 扩散。

    返回补全后的颜色、补全区置信度（按扩散距离衰减）、实际迭代次数。
    置信度上限 0.35，明确标记“不确定”，与真实观测样本区分。
    """
    out = color.copy()
    h, w = known.shape
    unknown = ~known
    if not np.any(unknown):
        return out, confidence, 0

    # 用边界已知颜色做初始均值，避免零值拖慢收敛
    if np.any(known):
        fill = out[known].mean(axis=0)
    else:
        fill = np.zeros(3)
    out[unknown] = fill

    kernel_sum = np.zeros((h, w))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            kernel_sum += np.roll(np.roll(known.astype(np.float64), dy, 0), dx, 1)
    has_known_neighbor = unknown & (kernel_sum > 0)

    iters = 0
    for it in range(max_iters):
        iters = it + 1
        acc = np.zeros_like(out)
        cnt = np.zeros((h, w))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                acc += np.roll(np.roll(out, dy, 0), dx, 1)
                cnt += 1
        cand = acc / np.maximum(cnt, 1)[..., None]
        out[unknown] = cand[unknown]
        if not np.any(unknown & ~np.isfinite(out).all(axis=-1)):
            pass
        # 全部未知格至少更新过一次即收敛（颜色不再为初始常数由迭代保证）
        if it >= max_iters:
            break

    # 置信度：与最近已知像素的曼哈顿距离衰减，封顶 0.35
    dist = np.full((h, w), max_iters, dtype=np.float64)
    dist[known] = 0.0
    for _ in range(max_iters):
        nd = dist.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nd = np.minimum(nd, np.roll(np.roll(dist, dy, 0), dx, 1) + 1)
        if np.allclose(nd, dist):
            break
        dist = nd
    inpaint_conf = np.clip(0.35 * np.exp(-dist / 6.0), 0.05, 0.35)
    new_conf = confidence.copy()
    new_conf[unknown] = inpaint_conf[unknown]
    return out, new_conf, iters


# ---------------------------------------------------------------------------
# 总装：中间帧
# ---------------------------------------------------------------------------


@dataclass
class InterpolationResult:
    frame: np.ndarray                 # uint8 HxWx3
    frame_float: np.ndarray           # float HxWx3
    t: float
    flow: FlowResult
    grid: SplatGrid
    holes: HoleMap
    confidence: np.ndarray            # HxW，[0,1]
    source_map: np.ndarray            # HxWx3：(src_x, src_y, src_side)，side -1=补全
    sample_source: np.ndarray         # HxW uint8，SampleSource 取值
    inpaint_iters: int
    scene_cut: SceneCutReport
    reprojection_table: dict         # 可验证的源目标坐标/时间比例抽样
    stats: dict = field(default_factory=dict)


def _build_reprojection_table(
    grid: SplatGrid,
    holes: HoleMap,
    flow01: np.ndarray,
    flow10: np.ndarray,
    t: float,
    samples: int = 16,
) -> dict:
    """抽取若干像素，给出完整可验证的重投影账目。"""
    h, w = grid.hit.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rng = np.random.default_rng(7)
    hit_idx = np.flatnonzero(grid.hit.ravel())
    rec_idx = np.flatnonzero(holes.recoverable.ravel())
    unp_idx = np.flatnonzero(holes.unrecoverable.ravel())

    def take(idx):
        if len(idx) == 0:
            return []
        pick = rng.choice(idx, size=min(samples, len(idx)), replace=False)
        rows = []
        for flat in pick:
            ty, tx = divmod(int(flat), w)
            side = int(grid.src_side[ty, tx])
            sx = float(grid.src_x[ty, tx])
            sy = float(grid.src_y[ty, tx])
            flow = flow01 if side == 0 else flow10
            ddx = float(flow[int(round(sy)), int(round(sx)), 0])
            ddy = float(flow[int(round(sy)), int(round(sx)), 1])
            scale = t if side == 0 else (1.0 - t)
            pred_x = sx + ddx * scale
            pred_y = sy + ddy * scale
            rows.append({
                "target_x": tx,
                "target_y": ty,
                "source_side": side,
                "source_x": round(sx, 3),
                "source_y": round(sy, 3),
                "flow_dx": round(ddx, 4),
                "flow_dy": round(ddy, 4),
                "time_scale": round(scale, 4),
                "predicted_target_x": round(pred_x, 3),
                "predicted_target_y": round(pred_y, 3),
                "rounding_error": round(
                    float(np.hypot(pred_x - tx, pred_y - ty)), 4
                ),
            })
        return rows

    rec_rows = []
    for flat in (rng.choice(rec_idx, size=min(samples, len(rec_idx)), replace=False)
                 if len(rec_idx) else []):
        ty, tx = divmod(int(flat), w)
        side = int(holes.recover_side[ty, tx])
        rec_rows.append({
            "target_x": tx,
            "target_y": ty,
            "source_side": side,
            "source_x": round(float(holes.recover_x[ty, tx]), 3),
            "source_y": round(float(holes.recover_y[ty, tx]), 3),
            "kind": "recoverable_extrapolation",
        })

    unp_rows = [
        {"target_x": int(xs.ravel()[i]), "target_y": int(ys.ravel()[i]),
         "kind": "unrecoverable_limited_inpaint"}
        for i in (rng.choice(unp_idx, size=min(samples, len(unp_idx)), replace=False)
                  if len(unp_idx) else [])
    ]
    return {
        "t": t,
        "hit_samples": take(hit_idx),
        "recoverable_samples": rec_rows,
        "unrecoverable_samples": unp_rows,
        "note": "predicted = source + flow * time_scale；可与 target 坐标逐项核对",
    }


def interpolate_frame(
    img0: np.ndarray,
    img1: np.ndarray,
    t: float = 0.5,
    block: int = 7,
    search: int = 12,
    inpaint_iters: int = 60,
    enforce_scene_cut: bool = True,
    flow: FlowResult | None = None,
) -> InterpolationResult:
    """合成 I0 与 I1 在时间 t∈[0,1] 处的中间帧。

    强制场景切换检查；检测到切换时抛 SceneCutError，不产出扭曲过渡帧。
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError("t 必须在 [0, 1] 区间")
    a = to_float_rgb(img0)
    b = to_float_rgb(img1)
    h, w = a.shape[:2]
    if b.shape[:2] != (h, w):
        raise ValueError("两帧尺寸必须一致")

    if flow is None:
        flow = estimate_bidirectional_flow(a, b, block=block, search=search)
    cut = detect_scene_cut(a, b, flow)
    if enforce_scene_cut and cut.is_cut:
        raise SceneCutError(
            f"检测到场景切换，拒绝插值：{cut.reason}"
        )

    # 两侧源帧上的几何遮挡掩码（前向投影多对一）
    occ0 = _source_occlusion_mask(flow.flow01, flow.consistent)
    # 侧 1 一致性：用反向 roundtrip 掩码（在 I1 网格）
    x10, y10 = _flow_to_coords(flow.flow10)
    in_b1 = (x10 >= 0) & (x10 < w) & (y10 >= 0) & (y10 < h)
    fwd_dx = bilinear_sample_gray(flow.flow01[..., 0], x10, y10)
    fwd_dy = bilinear_sample_gray(flow.flow01[..., 1], x10, y10)
    rt1 = np.sqrt(
        (flow.flow10[..., 0] + fwd_dx) ** 2 + (flow.flow10[..., 1] + fwd_dy) ** 2
    )
    consistent1 = in_b1 & np.isfinite(rt1) & (rt1 < 1.5)
    occ1 = _source_occlusion_mask(flow.flow10, consistent1)
    # 跨帧可见持续性：各自沿反向流投到另一帧，能匹配上表面者加分
    fg0 = _foreground_score(a, flow.flow01, flow.consistent, b, flow.flow10)
    fg1 = _foreground_score(b, flow.flow10, consistent1, a, flow.flow01)

    s0 = _splat_one_side(a, flow.flow01, t, 0, flow.consistent, h, w)
    s1 = _splat_one_side(b, flow.flow10, t, 1, consistent1, h, w)
    grid = _merge_splats(s0, s1, h, w, occ0, occ1, fg0, fg1)

    holes = classify_holes(grid, flow.flow01, flow.flow10, a, b, t)

    # 可回采区域：用外推源坐标双线性采样，并再次确认源坐标有效
    color = grid.color.copy()
    confidence = grid.confidence.copy()
    known = grid.hit.copy()
    source_side = grid.src_side.copy()
    src_x = grid.src_x.copy()
    src_y = grid.src_y.copy()

    rec = holes.recoverable
    if np.any(rec):
        from_side0 = rec & (holes.recover_side == 0)
        from_side1 = rec & (holes.recover_side == 1)
        for mask, img in ((from_side0, a), (from_side1, b)):
            if not np.any(mask):
                continue
            sampled = bilinear_sample(
                img, holes.recover_x[mask], holes.recover_y[mask]
            )
            ok = np.isfinite(sampled).all(axis=-1)
            yy, xx = np.where(mask)
            color[yy[ok], xx[ok]] = sampled[ok]
            confidence[yy[ok], xx[ok]] = 0.7
            known[yy[ok], xx[ok]] = True
            source_side[yy[ok], xx[ok]] = holes.recover_side[yy[ok], xx[ok]]
            src_x[yy[ok], xx[ok]] = holes.recover_x[yy[ok], xx[ok]]
            src_y[yy[ok], xx[ok]] = holes.recover_y[yy[ok], xx[ok]]
        # 采样失败的格降级为不可回采
        failed = rec & ~known
        holes.unrecoverable[failed] = True
        holes.recoverable[failed] = False

    # 待填区域：有限补全
    final_color, confidence, used_iters = limited_diffusion_inpaint(
        color, known, confidence, max_iters=inpaint_iters
    )

    sample_source = np.full((h, w), SampleSource.NONE.value, dtype=np.uint8)
    sample_source[grid.hit & (grid.n_hits_0 > 0) & (grid.n_hits_1 == 0)] = (
        SampleSource.SIDE0.value
    )
    sample_source[grid.hit & (grid.n_hits_1 > 0) & (grid.n_hits_0 == 0)] = (
        SampleSource.SIDE1.value
    )
    sample_source[grid.hit & (grid.n_hits_0 > 0) & (grid.n_hits_1 > 0)] = (
        SampleSource.BOTH.value
    )
    sample_source[holes.recoverable] = SampleSource.SIDE0.value  # 侧别另存
    sample_source[holes.unrecoverable] = SampleSource.INPAINT.value

    source_map = np.stack(
        [src_x, src_y, source_side.astype(np.float64)], axis=-1
    )

    stats = {
        "hit_ratio": float(np.mean(grid.hit)),
        "collision_ratio": float(np.mean(grid.collision)),
        "covered_ratio": float(np.mean(grid.covered)),
        "hole_ratio": float(np.mean(~grid.hit)),
        "recoverable_ratio": float(np.mean(holes.recoverable)),
        "unrecoverable_ratio": float(np.mean(holes.unrecoverable)),
        "inpaint_ratio": float(np.mean(holes.unrecoverable)),
        "inpaint_iters": used_iters,
        "mean_confidence": float(np.mean(confidence)),
        "consistency_ratio": flow.consistency_ratio,
    }

    table = _build_reprojection_table(grid, holes, flow.flow01, flow.flow10, t)

    return InterpolationResult(
        frame=to_uint8(final_color),
        frame_float=final_color,
        t=t,
        flow=flow,
        grid=grid,
        holes=holes,
        confidence=confidence,
        source_map=source_map,
        sample_source=sample_source,
        inpaint_iters=used_iters,
        scene_cut=cut,
        reprojection_table=table,
        stats=stats,
    )
