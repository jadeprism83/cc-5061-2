# 双向运动帧插值系统

依据双向运动场把 I0、I1 两侧像素前向映射到指定时刻 t，处理碰撞、空洞与遮挡，
合成中间帧。**不使用两图透明混合冒充运动补偿**；无法从两帧观测到的内容只做
明确的有限补全并在可信图标记为不确定。

## 目录

- `interp/core.py`：合成内核（块匹配双向流、前后向一致性、前向 splat、碰撞/遮挡
  裁决、空洞分类、有限补全、场景切换检测、重投影审计）。
- `interp/sequences.py`：带已知中间真值的合成序列（移动形状 / 交叉 / 出画 / 切镜）。
- `interp/viz.py`：运动、有效性、遮挡、空洞、可信图等分层可视化。
- `interp/metrics.py`：像素误差（MAE/PSNR）、边缘重影、空洞范围度量。
- `interp/encode.py`：H.264 mp4 编码，无 ffmpeg 时回退 GIF。
- `web/`：内核驱动的运动分析播放器（标准库 http.server，无前端框架）。
- `tests/run_evaluation.py`：三序列 × 三时刻评测 + 切镜拒绝 + 朴素混合对照。

## 启动运动分析播放器（界面）

```bash
cd /workspace/frame_interp
python3 -m web.app
```

然后浏览器打开 <http://127.0.0.1:8000/>。

界面能力（全部由 `interpolate_frame` 内核实时驱动）：

- 序列下拉：`moving_shapes` / `crossing` / `exit_frame` / `scene_cut`。
- t 滑块（0–1）、±0.05 步进、按真值网格（0.25）逐帧、自动播放。
- 结果区：中间帧 + 遮挡/补全叠加，以及运动、有效性、碰撞、空洞、可信图分层。
- 内核统计与“重投影审计表”（`predicted = source + flow × time_scale`，可逐项核对）。
- 选择 `scene_cut` 时内核抛 `SceneCutError`，拒绝原因直接回显到结果区，不生成
  扭曲过渡帧——失败反馈回中间帧界面。

## 跑量化评测

```bash
python3 tests/run_evaluation.py
```

输出在 `output/`：每序列每时刻的分层 PNG、真值对照、可播放 mp4，以及
`evaluation_report.json`（指标 + 重投影抽样）。

## 关键设计约束

- 对称交叉的两个同形物体，纯双向流不含深度信息，属本质歧义。系统只用几何证据
  （运动幅度、前后向一致性、跨帧持续性）裁决；仍并列时按参考帧 I0 的确定性层序
  选样，并把这些碰撞格可信度压到 ≤0.35，标记为不确定，不宣称恢复了真实隐藏细节。
- 两侧均未观测到的区域仅做有限次 Laplace 扩散补全，可信度封顶 0.35。
