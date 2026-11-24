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
- `interp/fusion.py`：五种中间帧融合原型（前向平均 / 分层择源 / 反向取样 /
  联合可见性 / 可见性切换），与 `core` 共用同一双向运动场。
- `interp/fusion_scenes.py`：同一轮廓扫过单色与细纹背景、交叉前景、显露背景。
- `interp/fusion_metrics.py`：错误遮盖、重影面积、空洞率、边缘误差（逐像素可复算）。
- `interp/fusion_viz.py`：融合原型的中间帧、遮挡叠加、选源侧、冲突分层。
- `web/`：内核驱动的运动分析播放器（标准库 http.server，无前端框架）。
- `tests/run_evaluation.py`：三序列 × 三时刻评测 + 切镜拒绝 + 朴素混合对照。
- `tests/run_fusion_experiment.py`：融合策略对照实验 + `--verify` 回读复算。

## 启动运动分析播放器（界面）

```bash
cd /workspace/frame_interp
python3 -m web.app
```

然后浏览器打开 <http://127.0.0.1:8000/>。这是中间帧的唯一图形入口，命令行脚本
（`tests/*.py`）只做批量复算与落盘，不替代界面交互。

界面能力（全部由内核实时驱动，图像与遮挡区域都来自实际输出）：

- 序列下拉分两组：
  - **融合实验场景**：`sweep_solid/texture`、`crossing_solid/texture`、
    `reveal_solid/texture`，进入五种融合策略逐方法对照；
  - **运动分析（旧样例）**：`moving_shapes` / `crossing` / `exit_frame` / `scene_cut`，
    走原有 `interpolate_frame` 路径。
- 融合场景下：多选对照方法，每个方法并排显示 **中间帧** 与 **遮挡区域叠加**，
  另给选源侧图、冲突/补全图；同屏显示真值帧与真值前景掩码。
- t 滑块（0–1）、±0.05 步进、按真值网格（0.25）逐帧、自动播放；移动 t 后逐帧
  对照中间图像与遮挡区域。
- **运行状态区**始终可见：就绪/拒绝/错误徽标、本次请求耗时；每个方法卡显示运行
  秒数、命中/空洞/冲突比例，并在指标越界时列出 **异常条目**（错误遮盖、漏遮、
  重影面积、两侧冲突、t 未对齐）。
- `scene_cut` 选择时内核抛 `SceneCutError`，拒绝原因回显到结果区，不生成扭曲过渡帧。
- 当 t 不在真值时间网格上时，图像仍是当前 t，指标徽标会标出“t 未对齐”并注明
  指标对应的最近真值帧。

## 跑量化评测

```bash
python3 tests/run_evaluation.py        # 旧样例：三序列 × 三时刻 + 切镜拒绝
python3 tests/run_fusion_experiment.py # 融合对照实验，见 FUSION_EXPERIMENT.md
python3 tests/run_fusion_experiment.py --verify  # 回读 JSON/PNG 复算，核对重影与空洞
```

输出在 `output/`：每序列每时刻的分层 PNG、真值对照、可播放 mp4，以及
`evaluation_report.json`（指标 + 重投影抽样）。融合对照的指标、掩码与结论见
`FUSION_EXPERIMENT.md` 与 `output/fusion_experiment/fusion_report.json`。

## 关键设计约束

- 对称交叉的两个同形物体，纯双向流不含深度信息，属本质歧义。系统只用几何证据
  （运动幅度、前后向一致性、跨帧持续性）裁决；仍并列时按参考帧 I0 的确定性层序
  选样，并把这些碰撞格可信度压到 ≤0.35，标记为不确定，不宣称恢复了真实隐藏细节。
- 两侧均未观测到的区域仅做有限次 Laplace 扩散补全，可信度封顶 0.35。
