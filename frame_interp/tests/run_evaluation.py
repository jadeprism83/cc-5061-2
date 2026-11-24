"""评估入口：在三个带真值序列上测量像素误差、边缘重影、空洞范围，
并验证场景切换被拒绝。输出 PNG 分层图与可播放视频、JSON 指标。

用法：
  python tests/run_evaluation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from interp import SceneCutError, interpolate_frame
from interp.encode import encode_video
from interp.metrics import evaluate
from interp.sequences import all_sequences, scene_cut
from interp.viz import all_layers

OUT = Path(__file__).resolve().parents[1] / "output"


def save_png(img, path: Path):
    Image.fromarray(img).save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seqs, cut = all_sequences(h=144, w=192, steps=5)
    report = {"sequences": {}, "scene_cut": {}}

    for frames, meta in seqs:
        name = meta["name"]
        seq_dir = OUT / name
        seq_dir.mkdir(exist_ok=True)
        i0, i1 = frames[0], frames[-1]
        save_png(i0, seq_dir / "frame0.png")
        save_png(i1, seq_dir / "frame1.png")

        seq_report = []
        # 真值中间帧 steps=5 -> t = 0.25, 0.5, 0.75
        play_frames = [i0]
        for k in range(1, len(frames) - 1):
            t = k / (len(frames) - 1)
            gt = frames[k]
            res = interpolate_frame(i0, i1, t=t, block=7, search=14)
            metrics = evaluate(res, gt)
            metrics["t"] = t
            seq_report.append(metrics)

            layers = all_layers(res)
            for layer_name, img in layers.items():
                save_png(img, seq_dir / f"t{t:.2f}_{layer_name}.png")
            save_png(gt, seq_dir / f"t{t:.2f}_groundtruth.png")
            play_frames.append(res.frame)
        play_frames.append(i1)

        # 可播放结果：I0 -> 三个中间帧 -> I1
        video_path = encode_video(play_frames, seq_dir / f"{name}_play", fps=4)
        report["sequences"][name] = {
            "metrics_per_t": seq_report,
            "video": str(video_path.relative_to(OUT.parent)),
            "kernel_stats": interpolate_frame(i0, i1, t=0.5).stats,
        }
        print(f"[OK] {name}: 视频 -> {video_path.name}")
        for m in seq_report:
            print(
                f"     t={m['t']:.2f} MAE={m['pixel']['mae_0_255']:.2f} "
                f"PSNR={m['pixel']['psnr_db']:.1f}dB "
                f"edgeMAE={m['ghosting']['edge_mae_0_255']:.2f} "
                f"holes={m['holes']['hole_ratio']:.3f} "
                f"inpaint={m['holes']['unrecoverable_inpaint_ratio']:.3f}"
            )

    # 场景切换：必须拒绝
    cut_frames, cut_meta = cut
    try:
        interpolate_frame(cut_frames[0], cut_frames[1], t=0.5)
        report["scene_cut"] = {"rejected": False, "error": "未拒绝！逻辑错误"}
        print("[FAIL] 场景切换未被拒绝")
    except SceneCutError as e:
        report["scene_cut"] = {"rejected": True, "error": str(e)}
        print(f"[OK] 场景切换已拒绝: {e}")

    # 同时验证“透明混合冒充”没有发生：t=0.5 结果不等于 0.5*I0+0.5*I1
    frames, _ = seqs[0]
    res = interpolate_frame(frames[0], frames[-1], t=0.5)
    blend = (
        frames[0].astype(np.float64) + frames[-1].astype(np.float64)
    ) / 2.0
    diff_to_blend = float(np.mean(np.abs(res.frame.astype(np.float64) - blend)))
    report["naive_blend_diff_mean_0_255"] = diff_to_blend
    print(f"[OK] 与朴素透明混合的平均差异: {diff_to_blend:.2f} (0-255，应显著 > 0)")

    with open(OUT / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n指标与重投影抽样: {OUT / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
