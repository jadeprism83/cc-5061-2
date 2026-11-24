"""中间帧融合原型：四条构造路径，全部共享同一双向运动场。

方法（用于对照实验，不改变 core 的默认行为）：
  warp_blend  双向前向扭曲后透明平均。不做遮挡处理——基线，会重影/错误遮盖。
  layered     先前向扭曲，再按层择源（复用 core 的 splat / 碰撞 / 遮挡 / 空洞
              分类 / 有限补全）。即当前生产内核的融合路径。
  backward    双向反向取样：在目标时刻用多起点不动点求两帧对应点，取流一致
              性高的一侧；不做两侧联合可见性判定。
  joint       反向取样 + 联合可见性：用两侧前向投影的碰撞胜负作为可见性证据，
              与流一致性、前景评分一起形成两侧权重，再选定源；同表面一致时
              取均值，不同表面冲突时明确标低可信。全流程不做跨表面颜色平均。

所有方法的空洞使用同一有限扩散补全，保证像素误差、重影、空洞率可比。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import (
    _flow_to_coords,
    _foreground_score,
    _merge_splats,
    _source_occlusion_mask,
    _splat_one_side,
    bilinear_sample,
    bilinear_sample_gray,
    classify_holes,
    limited_diffusion_inpaint,
    to_float_rgb,
    to_uint8,
)


# ---------------------------------------------------------------------------
# 公共小工具
# ---------------------------------------------------------------------------


def side1_consistency(flow) -> np.ndarray:
    """I1 网格上的前后向一致性掩码（与 core.interpolate_frame 相同）。"""
    h, w = flow.flow01.shape[:2]
    x10, y10 = _flow_to_coords(flow.flow10)
    in_b1 = (x10 >= 0) & (x10 < w) & (y10 >= 0) & (y10 < h)
    fwd_dx = bilinear_sample_gray(flow.flow01[..., 0], x10, y10)
    fwd_dy = bilinear_sample_gray(flow.flow01[..., 1], x10, y10)
    rt1 = np.sqrt(
        (flow.flow10[..., 0] + fwd_dx) ** 2 + (flow.flow10[..., 1] + fwd_dy) ** 2
    )
    return in_b1 & np.isfinite(rt1) & (rt1 < 1.5)


@dataclass
class FusionOutput:
    method: str
    t: float
    frame: np.ndarray                  # uint8 HxWx3
    frame_float: np.ndarray
    hit: np.ndarray                    # 有真实源样本（非补全）
    fill: np.ndarray                   # 由有限扩散补全的格
    conflict: np.ndarray               # 两侧样本冲突（不同表面争夺）
    uncertain: np.ndarray              # 低可信（冲突/本质歧义）
    side_choice: np.ndarray            # -1 补全 / 0 / 1 / 3 两侧一致
    confidence: np.ndarray
    runtime_s: float
    stats: dict = field(default_factory=dict)


METHOD_DESCRIPTIONS = {
    "warp_blend": "双向前向扭曲后平均（无遮挡处理）",
    "layered": "先前向扭曲，再按层择源（core 生产路径）",
    "backward": "反向取样，无两侧联合可见性",
    "joint": "反向取样 + 联合可见性与两侧权重后选定",
    "vis_switch": "按可见性切换：明确处反向样本，其余回退分层",
}


# ---------------------------------------------------------------------------
# 前向扭曲基线
# ---------------------------------------------------------------------------


def _forward_scatter(
    img: np.ndarray, flow: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """最近邻前向散射：返回目标格颜色与命中。同一格重复命中取最后写入。"""
    h, w = img.shape[:2]
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    tx = np.rint(xs + flow[..., 0] * scale).astype(np.int64)
    ty = np.rint(ys + flow[..., 1] * scale).astype(np.int64)
    inside = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    out = np.zeros((h, w, 3), dtype=np.float64)
    hit = np.zeros((h, w), dtype=bool)
    out[ty[inside], tx[inside]] = img[ys[inside], xs[inside]]
    hit[ty[inside], tx[inside]] = True
    return out, hit


def method_warp_blend(img0, img1, flow, t, inpaint_iters=60) -> FusionOutput:
    import time
    t0 = time.perf_counter()
    a, b = to_float_rgb(img0), to_float_rgb(img1)
    h, w = a.shape[:2]
    c0, h0 = _forward_scatter(a, flow.flow01, t)
    c1, h1 = _forward_scatter(b, flow.flow10, 1.0 - t)
    both = h0 & h1
    color = np.zeros_like(a)
    color[h0 & ~h1] = c0[h0 & ~h1]
    color[h1 & ~h0] = c1[h1 & ~h0]
    color[both] = 0.5 * (c0[both] + c1[both])
    hit = h0 | h1
    fill_raw = ~hit
    color, confidence, iters = limited_diffusion_inpaint(
        color, hit, np.where(hit, 0.9, 0.0), max_iters=inpaint_iters
    )
    fill = fill_raw
    side_choice = np.full((h, w), -1, dtype=np.int8)
    side_choice[h0 & ~h1] = 0
    side_choice[h1 & ~h0] = 1
    side_choice[both] = 3
    stats = {
        "method": "warp_blend",
        "hit_ratio": float(hit.mean()),
        "raw_hole_ratio": float(fill_raw.mean()),
        "fill_ratio": float(fill.mean()),
        "conflict_ratio": float(both.mean()),
        "cross_side_agreement_ratio": float(both.mean()),
        "uncertain_ratio": float(both.mean()),
        "mean_confidence": float(confidence.mean()),
        "inpaint_iters": int(iters),
    }
    # 基线没有遮挡裁决，两侧同时命中一律透明平均；conflict 掩码留空，
    # 避免把“同一表面被两侧共同观测”误报成两侧冲突。
    no_conflict = np.zeros((h, w), dtype=bool)
    return FusionOutput(
        "warp_blend", t, to_uint8(color), color, hit, fill, no_conflict,
        no_conflict, side_choice, confidence,
        time.perf_counter() - t0, stats,
    )


# ---------------------------------------------------------------------------
# 分层融合（core 生产路径）
# ---------------------------------------------------------------------------


def _layered_grid(img0, img1, flow, t):
    a, b = to_float_rgb(img0), to_float_rgb(img1)
    h, w = a.shape[:2]
    consistent1 = side1_consistency(flow)
    occ0 = _source_occlusion_mask(flow.flow01, flow.consistent)
    occ1 = _source_occlusion_mask(flow.flow10, consistent1)
    fg0 = _foreground_score(a, flow.flow01, flow.consistent, b, flow.flow10)
    fg1 = _foreground_score(b, flow.flow10, consistent1, a, flow.flow01)
    s0 = _splat_one_side(a, flow.flow01, t, 0, flow.consistent, h, w)
    s1 = _splat_one_side(b, flow.flow10, t, 1, consistent1, h, w)
    grid = _merge_splats(s0, s1, h, w, occ0, occ1, fg0, fg1)
    holes = classify_holes(grid, flow.flow01, flow.flow10, a, b, t)
    return a, b, grid, holes


def method_layered(img0, img1, flow, t, inpaint_iters=60) -> FusionOutput:
    import time
    t0 = time.perf_counter()
    a, b, grid, holes = _layered_grid(img0, img1, flow, t)
    h, w = a.shape[:2]

    color = grid.color.copy()
    confidence = grid.confidence.copy()
    known = grid.hit.copy()
    side_choice = np.full((h, w), -1, dtype=np.int8)
    side_choice[grid.hit] = grid.src_side[grid.hit].astype(np.int8)

    rec = holes.recoverable
    if np.any(rec):
        for side, img in ((0, a), (1, b)):
            mask = rec & (holes.recover_side == side)
            if not np.any(mask):
                continue
            sampled = bilinear_sample(img, holes.recover_x[mask], holes.recover_y[mask])
            ok = np.isfinite(sampled).all(axis=-1)
            yy, xx = np.where(mask)
            color[yy[ok], xx[ok]] = sampled[ok]
            confidence[yy[ok], xx[ok]] = 0.7
            known[yy[ok], xx[ok]] = True
            side_choice[yy[ok], xx[ok]] = side
    unresolved = ~known
    color, confidence, iters = limited_diffusion_inpaint(
        color, known, confidence, max_iters=inpaint_iters
    )
    fill = unresolved
    conflict = grid.collision & grid.hit
    uncertain = conflict | (confidence < 0.5)
    stats = {
        "method": "layered",
        "hit_ratio": float(grid.hit.mean()),
        "raw_hole_ratio": float((~grid.hit).mean()),
        "recoverable_ratio": float(holes.recoverable.mean()),
        "fill_ratio": float(fill.mean()),
        "conflict_ratio": float(conflict.mean()),
        "uncertain_ratio": float(uncertain.mean()),
        "mean_confidence": float(confidence.mean()),
        "inpaint_iters": int(iters),
    }
    return FusionOutput(
        "layered", t, to_uint8(color), color, grid.hit, fill, conflict,
        uncertain, side_choice, confidence, time.perf_counter() - t0, stats,
    )


# ---------------------------------------------------------------------------
# 反向取样（多起点不动点）
# ---------------------------------------------------------------------------


def _flow_start_vectors(flow: np.ndarray, k: int = 10) -> np.ndarray:
    """从运动场中抽取出现频率最高的一组位移向量，作为不动点迭代起点。

    反向取样必须独立于前向 splat，因此起点取自全场运动的离散取值分布，
    并强制包含零位移（静态背景）。
    """
    q = np.rint(flow.reshape(-1, 2)).astype(np.int64)
    uniq, counts = np.unique(q, axis=0, return_counts=True)
    order = np.argsort(-counts)[:k]
    vecs = [tuple(int(v) for v in uniq[i]) for i in order]
    vecs.append((0, 0))
    # 最大位移方向也纳入，避免大运动表面被频率屏蔽
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    iy, ix = np.unravel_index(int(np.argmax(mag)), mag.shape)
    vecs.append((int(round(flow[iy, ix, 0])), int(round(flow[iy, ix, 1]))))
    return np.array(sorted(set(vecs)), dtype=np.float64)


def _backward_candidates(
    img: np.ndarray,
    flow: np.ndarray,
    scale: float,
    starts: np.ndarray,
    iters: int = 8,
) -> dict:
    """对每个目标像素求 x_s = x_t - scale * flow_s(x_s) 的全部收敛解。

    返回每个目标格的：最佳源坐标/颜色/是否有效/流一致性/闭环误差，以及
    该侧在目标格是否存在第二个不同解（多表面竞争）。
    """
    h, w = img.shape[:2]
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    tx = xs.astype(np.float64)
    ty = ys.astype(np.float64)
    fx, fy = flow[..., 0], flow[..., 1]

    best_score = np.full((h, w), -1e9)
    best_x = np.full((h, w), np.nan)
    best_y = np.full((h, w), np.nan)
    best_valid = np.zeros((h, w), dtype=bool)
    second_x = np.full((h, w), np.nan)
    second_y = np.full((h, w), np.nan)
    second_score = np.full((h, w), -1e9)

    for vx, vy in starts:
        sx = tx - scale * vx
        sy = ty - scale * vy
        for _ in range(iters):
            dx = bilinear_sample_gray(fx, sx, sy)
            dy = bilinear_sample_gray(fy, sx, sy)
            sx_new = tx - scale * dx
            sy_new = ty - scale * dy
            sx = np.where(np.isfinite(sx_new), sx_new, np.nan)
            sy = np.where(np.isfinite(sy_new), sy_new, np.nan)
        dx = bilinear_sample_gray(fx, sx, sy)
        dy = bilinear_sample_gray(fy, sx, sy)
        proj_x = sx + scale * dx
        proj_y = sy + scale * dy
        with np.errstate(invalid="ignore"):
            err = np.abs(proj_x - tx) + np.abs(proj_y - ty)
        valid = (
            np.isfinite(sx) & np.isfinite(sy) & np.isfinite(err)
            & (err < 1.5)
            & (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        )
        score = np.where(valid, 2.0 - err, -1e9)
        better = score > best_score
        # 旧最佳下移为第二解
        second_x = np.where(better & best_valid, best_x, second_x)
        second_y = np.where(better & best_valid, best_y, second_y)
        second_score = np.where(better & best_valid, best_score, second_score)
        best_x = np.where(better, sx, best_x)
        best_y = np.where(better, sy, best_y)
        best_score = np.where(better, score, best_score)
        best_valid = best_valid | valid
        # 非最佳但与最佳解相距较远的另一收敛解 -> 同侧多表面
        far = (
            valid & best_valid & (best_score >= score)
            & (np.abs(sx - best_x) + np.abs(sy - best_y) > 2.0)
            & (score > second_score)
        )
        second_x = np.where(far, sx, second_x)
        second_y = np.where(far, sy, second_y)
        second_score = np.where(far, score, second_score)

    sx_c = np.clip(np.nan_to_num(best_x, nan=0.0), 0, w - 1)
    sy_c = np.clip(np.nan_to_num(best_y, nan=0.0), 0, h - 1)
    color = bilinear_sample(img, best_x, best_y)
    valid = best_valid & np.isfinite(color).all(axis=-1)
    second_valid = (
        np.isfinite(second_x) & np.isfinite(second_y)
        & (np.abs(second_x - best_x) + np.abs(second_y - best_y) > 2.0)
    )
    return {
        "valid": valid,
        "x": best_x,
        "y": best_y,
        "color": color,
        "err": np.where(valid, best_score, np.inf),
        "src_ix": sx_c.astype(int),
        "src_iy": sy_c.astype(int),
        "second_valid": second_valid,
        "second_x": second_x,
        "second_y": second_y,
    }


def _backward_base(img0, img1, flow, t, inpaint_iters=60):
    a, b = to_float_rgb(img0), to_float_rgb(img1)
    h, w = a.shape[:2]
    starts01 = _flow_start_vectors(flow.flow01)
    starts10 = _flow_start_vectors(flow.flow10)
    c0 = _backward_candidates(a, flow.flow01, t, starts01)
    c1 = _backward_candidates(b, flow.flow10, 1.0 - t, starts10)
    return a, b, c0, c1, h, w


def method_backward(img0, img1, flow, t, inpaint_iters=60) -> FusionOutput:
    import time
    t0 = time.perf_counter()
    a, b, c0, c1, h, w = _backward_base(img0, img1, flow, t, inpaint_iters)
    consistent1 = side1_consistency(flow)
    # 无联合可见性：只按各自闭环误差与流一致性选一侧
    conf0 = np.where(flow.consistent[c0["src_iy"], c0["src_ix"]], 1.0, 0.45)
    conf1 = np.where(consistent1[c1["src_iy"], c1["src_ix"]], 1.0, 0.45)
    score0 = c0["valid"] * (conf0 + 0.02 * np.where(np.isfinite(c0["err"]), -c0["err"], -5))
    score1 = c1["valid"] * (conf1 + 0.02 * np.where(np.isfinite(c1["err"]), -c1["err"], -5))
    both = c0["valid"] & c1["valid"]
    color_diff = np.linalg.norm(
        np.nan_to_num(c0["color"]) - np.nan_to_num(c1["color"]), axis=-1
    )
    agree = both & (color_diff < 0.18)
    clash = both & ~agree
    take1 = (~c0["valid"] & c1["valid"]) | (clash & (score1 > score0))
    take0 = (c0["valid"] & ~c1["valid"]) | (clash & ~(score1 > score0))
    color = np.zeros((h, w, 3))
    color[take0] = c0["color"][take0]
    color[take1] = c1["color"][take1]
    color[agree] = 0.5 * (c0["color"][agree] + c1["color"][agree])
    hit = c0["valid"] | c1["valid"]
    confidence = np.zeros((h, w))
    confidence[take0] = conf0[take0] * 0.8
    confidence[take1] = conf1[take1] * 0.8
    confidence[agree] = np.maximum(conf0[agree], conf1[agree])
    color, confidence, iters = limited_diffusion_inpaint(
        color, hit, confidence, max_iters=inpaint_iters
    )
    side_choice = np.full((h, w), -1, dtype=np.int8)
    side_choice[take0] = 0
    side_choice[take1] = 1
    side_choice[agree] = 3
    fill = ~hit
    stats = {
        "method": "backward",
        "hit_ratio": float(hit.mean()),
        "raw_hole_ratio": float(fill.mean()),
        "fill_ratio": float(fill.mean()),
        "conflict_ratio": float(clash.mean()),
        "uncertain_ratio": float(clash.mean()),
        "mean_confidence": float(confidence.mean()),
        "inpaint_iters": int(iters),
    }
    return FusionOutput(
        "backward", t, to_uint8(color), color, hit, fill, clash,
        clash, side_choice, confidence, time.perf_counter() - t0, stats,
    )


# ---------------------------------------------------------------------------
# 联合可见性 + 两侧权重
# ---------------------------------------------------------------------------


def _forward_collision_winner(img0, img1, flow, t, fg0, fg1) -> dict:
    """两侧前向投影在中点的碰撞胜负，作为反向候选的可见性证据。

    返回每个目标格的：是否有碰撞、胜出侧、两侧是否都有前向命中。
    """
    a, b = to_float_rgb(img0), to_float_rgb(img1)
    h, w = a.shape[:2]
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    tx0 = np.rint(xs + flow["f01"][..., 0] * t).astype(np.int64)
    ty0 = np.rint(ys + flow["f01"][..., 1] * t).astype(np.int64)
    tx1 = np.rint(xs + flow["f10"][..., 0] * (1.0 - t)).astype(np.int64)
    ty1 = np.rint(ys + flow["f10"][..., 1] * (1.0 - t)).astype(np.int64)
    in0 = (tx0 >= 0) & (tx0 < w) & (ty0 >= 0) & (ty0 < h)
    in1 = (tx1 >= 0) & (tx1 < w) & (ty1 >= 0) & (ty1 < h)

    hit0 = np.zeros((h, w), dtype=bool)
    hit1 = np.zeros((h, w), dtype=bool)
    fgmap0 = np.full((h, w), -np.inf)
    fgmap1 = np.full((h, w), -np.inf)
    hit0[ty0[in0], tx0[in0]] = True
    hit1[ty1[in1], tx1[in1]] = True
    np.maximum.at(fgmap0, (ty0[in0], tx0[in0]), fg0[ys[in0], xs[in0]])
    np.maximum.at(fgmap1, (ty1[in1], tx1[in1]), fg1[ys[in1], xs[in1]])
    collide = hit0 & hit1
    winner = np.full((h, w), -1, dtype=np.int8)
    winner[collide & (fgmap1 > fgmap0 + 1e-6)] = 1
    winner[collide & ~(fgmap1 > fgmap0 + 1e-6)] = 0
    return {"hit0": hit0, "hit1": hit1, "collide": collide, "winner": winner}


def method_joint(img0, img1, flow, t, inpaint_iters=60) -> FusionOutput:
    import time
    t0 = time.perf_counter()
    a, b = to_float_rgb(img0), to_float_rgb(img1)
    h, w = a.shape[:2]
    consistent1 = side1_consistency(flow)
    fg0 = _foreground_score(a, flow.flow01, flow.consistent, b, flow.flow10)
    fg1 = _foreground_score(b, flow.flow10, consistent1, a, flow.flow01)
    vis = _forward_collision_winner(
        img0, img1, {"f01": flow.flow01, "f10": flow.flow10}, t, fg0, fg1
    )

    starts01 = _flow_start_vectors(flow.flow01)
    starts10 = _flow_start_vectors(flow.flow10)
    c0 = _backward_candidates(a, flow.flow01, t, starts01)
    c1 = _backward_candidates(b, flow.flow10, 1.0 - t, starts10)

    # 逐候选的联合权重：流一致性 × 可见性 × 归一化前景评分
    def weights(c, consistent, side):
        cons = np.where(consistent[c["src_iy"], c["src_ix"]], 1.0, 0.45)
        fgn = np.minimum(fg0[c["src_iy"], c["src_ix"]], 4.0) / 4.0 \
            if side == 0 else np.minimum(fg1[c["src_iy"], c["src_ix"]], 4.0) / 4.0
        vis_map = np.full((h, w), 0.7)
        vis_map[vis["winner"] == side] = 1.0
        vis_map[vis["collide"] & (vis["winner"] != side)] = 0.2
        vis_map[~vis["collide"] & (vis["hit0"] | vis["hit1"])] = 0.9
        score = cons * vis_map * (0.6 + 0.4 * fgn)
        return score, cons, vis_map

    w0, cons0, vis0 = weights(c0, flow.consistent, 0)
    w1, cons1, vis1 = weights(c1, consistent1, 1)
    w0 = np.where(c0["valid"], w0, 0.0)
    w1 = np.where(c1["valid"], w1, 0.0)

    both = c0["valid"] & c1["valid"]
    diff = np.linalg.norm(
        np.nan_to_num(c0["color"]) - np.nan_to_num(c1["color"]), axis=-1
    )
    agree = both & (diff < 0.18)
    clash = both & ~agree
    prefer1 = clash & (w1 > w0 + 1e-9)
    prefer0 = clash & ~prefer1
    # 权重完全并列：确定性保留侧 0，并标为不确定（双向流无深度信息）
    tie = clash & (np.abs(w1 - w0) <= 1e-9)

    only0 = c0["valid"] & ~c1["valid"]
    only1 = c1["valid"] & ~c0["valid"]
    take0 = only0 | prefer0
    take1 = only1 | prefer1

    color = np.zeros((h, w, 3))
    color[take0] = c0["color"][take0]
    color[take1] = c1["color"][take1]
    color[agree] = 0.5 * (c0["color"][agree] + c1["color"][agree])
    confidence = np.zeros((h, w))
    confidence[take0] = np.minimum(w0[take0], 1.0)
    confidence[take1] = np.minimum(w1[take1], 1.0)
    confidence[agree] = np.maximum(w0[agree], w1[agree])
    # 单侧样本若源在另一帧的投影中被判为“输家”，其可见性已压低到 0.2，
    # 这里不额外改写；冲突与并列统一封顶
    confidence[clash] = np.minimum(confidence[clash], 0.45)
    confidence[tie] = np.minimum(confidence[tie], 0.35)

    hit = c0["valid"] | c1["valid"]
    fill = ~hit
    color, confidence, iters = limited_diffusion_inpaint(
        color, hit, confidence, max_iters=inpaint_iters
    )
    side_choice = np.full((h, w), -1, dtype=np.int8)
    side_choice[take0] = 0
    side_choice[take1] = 1
    side_choice[agree] = 3
    uncertain = clash | fill | (confidence < 0.4)
    stats = {
        "method": "joint",
        "hit_ratio": float(hit.mean()),
        "raw_hole_ratio": float(fill.mean()),
        "fill_ratio": float(fill.mean()),
        "conflict_ratio": float(clash.mean()),
        "tie_ratio": float(tie.mean()),
        "uncertain_ratio": float(uncertain.mean()),
        "mean_confidence": float(confidence.mean()),
        "inpaint_iters": int(iters),
    }
    return FusionOutput(
        "joint", t, to_uint8(color), color, hit, fill, clash,
        uncertain, side_choice, confidence, time.perf_counter() - t0, stats,
    )


# ---------------------------------------------------------------------------
# 按可见性切换：可见性明确处采用反向样本，其余回退分层择源
# ---------------------------------------------------------------------------


def method_vis_switch(img0, img1, flow, t, inpaint_iters=60) -> FusionOutput:
    """可见性驱动的切换策略。

    规则（逐像素）：
      1. 两侧反向解都有效且同色 -> 同一表面，取均值（与 warp_blend 在安全区一致）；
      2. 仅一侧反向解有效，且该侧在前向碰撞中未被判为遮挡方 -> 采用该侧；
      3. 其余（冲突、多解、两侧都不可见）-> 回退分层融合的择源结果。
    空洞与补全沿用分层路径，保证可比。
    """
    import time
    t0 = time.perf_counter()
    a, b = to_float_rgb(img0), to_float_rgb(img1)
    h, w = a.shape[:2]
    consistent1 = side1_consistency(flow)
    fg0 = _foreground_score(a, flow.flow01, flow.consistent, b, flow.flow10)
    fg1 = _foreground_score(b, flow.flow10, consistent1, a, flow.flow01)
    vis = _forward_collision_winner(
        img0, img1, {"f01": flow.flow01, "f10": flow.flow10}, t, fg0, fg1
    )
    starts01 = _flow_start_vectors(flow.flow01)
    starts10 = _flow_start_vectors(flow.flow10)
    c0 = _backward_candidates(a, flow.flow01, t, starts01)
    c1 = _backward_candidates(b, flow.flow10, 1.0 - t, starts10)

    # 分层回退结果
    la, lb, grid, holes = _layered_grid(img0, img1, flow, t)

    diff = np.linalg.norm(
        np.nan_to_num(c0["color"]) - np.nan_to_num(c1["color"]), axis=-1
    )
    both = c0["valid"] & c1["valid"]
    agree = both & (diff < 0.18)
    vis0_ok = vis["winner"] != 1
    vis1_ok = vis["winner"] != 0
    only0 = c0["valid"] & ~c1["valid"] & vis0_ok
    only1 = c1["valid"] & ~c0["valid"] & vis1_ok
    use_backward = agree | only0 | only1

    color = grid.color.copy()
    side_choice = np.full((h, w), -1, dtype=np.int8)
    side_choice[grid.hit] = grid.src_side[grid.hit].astype(np.int8)
    confidence = grid.confidence.copy()

    color[agree] = 0.5 * (c0["color"][agree] + c1["color"][agree])
    side_choice[agree] = 3
    confidence[agree] = np.maximum(
        np.where(flow.consistent[c0["src_iy"][agree], c0["src_ix"][agree]], 1.0, 0.45),
        np.where(consistent1[c1["src_iy"][agree], c1["src_ix"][agree]], 1.0, 0.45),
    )
    color[only0] = c0["color"][only0]
    side_choice[only0] = 0
    confidence[only0] = 0.9
    color[only1] = c1["color"][only1]
    side_choice[only1] = 1
    confidence[only1] = 0.9

    rec = holes.recoverable & ~use_backward
    known = grid.hit | use_backward
    if np.any(rec):
        for side, img in ((0, a), (1, b)):
            mask = rec & (holes.recover_side == side)
            if not np.any(mask):
                continue
            sampled = bilinear_sample(img, holes.recover_x[mask], holes.recover_y[mask])
            ok = np.isfinite(sampled).all(axis=-1)
            yy, xx = np.where(mask)
            color[yy[ok], xx[ok]] = sampled[ok]
            known[yy[ok], xx[ok]] = True
            side_choice[yy[ok], xx[ok]] = side
    fill = ~known
    color, confidence, iters = limited_diffusion_inpaint(
        color, known, confidence, max_iters=inpaint_iters
    )
    clash = both & ~agree
    uncertain = clash | (confidence < 0.5) | fill
    stats = {
        "method": "vis_switch",
        "hit_ratio": float(known.mean()),
        "raw_hole_ratio": float(fill.mean()),
        "fill_ratio": float(fill.mean()),
        "conflict_ratio": float(clash.mean()),
        "uncertain_ratio": float(uncertain.mean()),
        "switched_to_backward_ratio": float(use_backward.mean()),
        "mean_confidence": float(confidence.mean()),
        "inpaint_iters": int(iters),
    }
    return FusionOutput(
        "vis_switch", t, to_uint8(color), color, known, fill, clash,
        uncertain, side_choice, confidence, time.perf_counter() - t0, stats,
    )


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


RUNNERS = {
    "warp_blend": method_warp_blend,
    "layered": method_layered,
    "backward": method_backward,
    "joint": method_joint,
    "vis_switch": method_vis_switch,
}


def run_method(name: str, img0, img1, t: float, flow=None,
               inpaint_iters: int = 60) -> FusionOutput:
    """按名称运行一种融合构造；flow 缺省时现估（实验脚本会复用同一 flow）。"""
    if name not in RUNNERS:
        raise ValueError(f"未知融合方法 {name}；可用：{sorted(RUNNERS)}")
    if flow is None:
        from .core import estimate_bidirectional_flow
        flow = estimate_bidirectional_flow(img0, img1)
    return RUNNERS[name](img0, img1, flow, t, inpaint_iters=inpaint_iters)
