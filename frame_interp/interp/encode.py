"""把帧序列编码为可播放文件：优先 H.264 mp4，无 ffmpeg 时回退 GIF。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def _ffmpeg_path() -> str | None:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def encode_mp4(
    frames: list[np.ndarray],
    out_path: str | Path,
    fps: int = 8,
) -> Path:
    """编码 H.264 yuv420p mp4，浏览器与系统播放器均可直接播放。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg_path()
    if ff is None:
        raise RuntimeError("未找到 ffmpeg，可改用 encode_gif")
    h, w = frames[0].shape[:2]
    cmd = [
        ff, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "medium",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.run(
        cmd, input=b"".join(f.tobytes() for f in frames),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 编码失败: {proc.stderr.decode('utf-8', 'ignore')[-500:]}"
        )
    return out_path


def encode_gif(
    frames: list[np.ndarray],
    out_path: str | Path,
    fps: int = 8,
) -> Path:
    """无 ffmpeg 时的回退方案：Pillow GIF。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    duration = int(round(1000 / fps))
    imgs[0].save(
        out_path, save_all=True, append_images=imgs[1:],
        duration=duration, loop=0, optimize=True,
    )
    return out_path


def encode_video(frames, out_path, fps=8) -> Path:
    out_path = Path(out_path)
    ff = _ffmpeg_path()
    if ff is not None:
        try:
            return encode_mp4(frames, out_path.with_suffix(".mp4"), fps)
        except RuntimeError:
            pass
    return encode_gif(frames, out_path.with_suffix(".gif"), fps)
