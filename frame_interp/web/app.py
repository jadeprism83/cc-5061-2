"""内核驱动的运动分析播放器后端（仅用 Python 标准库）。

所有中间帧、分层图、可信图、重投影抽样都来自 interp.interpolate_frame 这一个
合成内核，本服务不做任何独立的图像处理，只负责把内核结果编码给浏览器。

启动：
  python -m web.app
然后浏览器打开 http://127.0.0.1:8000/
"""

from __future__ import annotations

import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp import SceneCutError, interpolate_frame  # noqa: E402
from interp.sequences import all_sequences, scene_cut  # noqa: E402
from interp.viz import all_layers  # noqa: E402

HOST = "127.0.0.1"
PORT = 8000

# 进程级缓存：序列帧只构建一次；双向流按 (序列) 缓存，t 变化时复用同一运动估计，
# 保证滑块拖动时“运动分析”固定、只改变时间投影位置——真正由内核对 t 做出响应。
_SEQ_CACHE: dict[str, list] = {}
_FLOW_CACHE: dict[str, object] = {}
# 同一次 (序列, t) 的内核产物缓存，元数据与各分层端点共享，避免重复合成。
_RESULT_CACHE: dict[tuple, dict] = {}
_RESULT_CACHE_MAX = 8


def _load_sequences():
    if _SEQ_CACHE:
        return
    seqs, cut = all_sequences(h=144, w=192, steps=5)
    for frames, meta in seqs:
        _SEQ_CACHE[meta["name"]] = frames
    cut_frames, cut_meta = scene_cut(144, 192)
    _SEQ_CACHE[cut_meta["name"]] = cut_frames


def _png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _kernel_compute(seq_name: str, t: float) -> dict:
    """调用合成内核并整理浏览器需要的全部产物（带结果缓存）。"""
    key = (seq_name, round(float(t), 4))
    cached = _RESULT_CACHE.get(key)
    if cached is not None:
        return cached

    frames = _SEQ_CACHE[seq_name]
    img0, img1 = frames[0], frames[-1]
    flow = _FLOW_CACHE.get(seq_name)
    result = interpolate_frame(img0, img1, t=t, block=7, search=14, flow=flow)
    # 首次成功后缓存该序列的双向流（场景切换序列会先抛错，不会写入）
    _FLOW_CACHE.setdefault(seq_name, result.flow)

    layers = all_layers(result)
    images = {name: _png_bytes(img) for name, img in layers.items()}
    # 最近的已知真值帧（仅用于旁注对比，不参与合成）
    gt_index = min(range(len(frames)), key=lambda k: abs(k / (len(frames) - 1) - t))
    images["groundtruth"] = _png_bytes(frames[gt_index])
    images["frame0"] = _png_bytes(img0)
    images["frame1"] = _png_bytes(img1)

    data = {
        "ok": True,
        "t": t,
        "images": images,
        "stats": result.stats,
        "reprojection": result.reprojection_table,
        "gt_t": gt_index / (len(frames) - 1),
        "scene_cut": False,
    }
    if len(_RESULT_CACHE) >= _RESULT_CACHE_MAX:
        _RESULT_CACHE.pop(next(iter(_RESULT_CACHE)))
    _RESULT_CACHE[key] = data
    return data


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 静默，保持控制台干净
        pass

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_file(ROOT / "web" / "static" / "index.html", "text/html")
            return

        if path == "/api/sequences":
            _load_sequences()
            names = list(_SEQ_CACHE.keys())
            self._send_json({"sequences": names})
            return

        if path == "/api/compute":
            qs = parse_qs(parsed.query)
            try:
                seq_name = qs["seq"][0]
                t = float(qs["t"][0])
            except (KeyError, IndexError, ValueError):
                self._send_json({"ok": False, "error": "参数 seq / t 无效"}, 400)
                return
            _load_sequences()
            if seq_name not in _SEQ_CACHE:
                self._send_json({"ok": False, "error": f"未知序列 {seq_name}"}, 404)
                return
            t = min(1.0, max(0.0, t))
            try:
                data = _kernel_compute(seq_name, t)
            except SceneCutError as exc:
                # 场景切换：把内核的拒绝原因回灌到前端结果区，不产出扭曲过渡帧
                self._send_json({
                    "ok": False,
                    "scene_cut": True,
                    "error": str(exc),
                    "t": t,
                }, 200)
                return
            # 图像用 multipart 之外的简单方案：先返回 JSON 元数据 + 各层 data URL 体积
            # 为避免 base64 膨胀到 JSON 里，这里直接把 PNG 以独立端点按需取。
            self._send_json({
                "ok": True,
                "scene_cut": False,
                "t": t,
                "layers": list(data["images"].keys()),
                "stats": data["stats"],
                "reprojection": data["reprojection"],
                "gt_t": data["gt_t"],
            })
            return

        if path == "/api/layer":
            qs = parse_qs(parsed.query)
            try:
                seq_name = qs["seq"][0]
                t = float(qs["t"][0])
                layer = qs["layer"][0]
            except (KeyError, IndexError, ValueError):
                self._send_json({"ok": False, "error": "参数无效"}, 400)
                return
            _load_sequences()
            t = min(1.0, max(0.0, t))
            try:
                data = _kernel_compute(seq_name, t)
            except SceneCutError as exc:
                self._send_json({"ok": False, "scene_cut": True,
                                 "error": str(exc)}, 200)
                return
            img = data["images"].get(layer)
            if img is None:
                self.send_error(404, "unknown layer")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(img)))
            self.end_headers()
            self.wfile.write(img)
            return

        self.send_error(404, "not found")

    def _serve_file(self, fpath: Path, ctype: str):
        if not fpath.exists():
            self.send_error(404)
            return
        body = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    _load_sequences()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"运动分析播放器已启动: http://{HOST}:{PORT}/")
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
        server.shutdown()


if __name__ == "__main__":
    main()
