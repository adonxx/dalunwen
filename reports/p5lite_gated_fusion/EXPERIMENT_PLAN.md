# P5-lite 语义反馈与空间门控：50轮筛选方案

## 研究问题

三随机种子核心复现表明，当前完整 Drone-YOLO（MF-FPN + LSCD + Inner-WIoU）相较 YOLOv11s 对小目标有稳定收益，但 Large AP 平均降低 2.78 个百分点。此前将 P5 直接作为第四个检测尺度的消融在 50 轮下未通过：虽可能补充部分小目标信息，却使 Large AP 明显下降。因此，本轮不增加 P5 检测头，而检验：**能否把轻量化 P5 语义作为 P4 的条件信息，以空间门控方式向 P3/P2 反馈，同时避免损害大目标。**

## 候选结构

候选在完整 Drone-YOLO 的最深层 P4（40×40）之后插入 `P5LiteSpatialGate`：

```text
P4 (40×40, 256c)
  └─ depthwise stride-2 + 1×1 降维 + depthwise 3×3
       → P5-lite (20×20, 128c)
       → 上采样 + 1×1 投影到 P4 通道
P4 与投影语义特征 → 1×1 像素级空间门控 → 残差回写 P4
       → 原有 top-down P3/P2 与 bottom-up P4 路径 → 检测 P2/P3/P4
```

- P5-lite 只在融合内部产生，**不作为第四个预测层**；检测尺度仍是 P2/P3/P4，对应 stride 4/8/16。
- 门控是逐像素的一通道掩码，初始偏置为 -2（初值约为 0.12），使新增分支在训练初期近似保守残差，降低随机初始化特征污染原 MF-FPN 的风险。
- 未训练结构审计：2.843M 参数、19.33 GFLOPs（640），比 Full 对照 2.772M / 19.19 GFLOPs 仅增加 0.071M / 0.14 GFLOPs。

## 固定协议与公平对照

- 数据与评估：VisDrone2019，640×640；标准验证与 COCO 原图像素面积分桶（small < 32²、medium 32²–96²、large ≥ 96²）。
- 初始化、训练：`yolo11s.pt`、seed=0、SGD、50 epochs；除候选模块外，配置与已完成 `loss_inner_wiou_50e` 一致。
- 对照：`reports/nwd_loss_ablation/metrics/inner_wiou_{val,size_val}.json` 所记录的完整 Drone-YOLO 50轮结果：mAP50-95 20.63%、Small AP 12.93%、Medium AP 30.38%、Large AP 35.20%。
- 推理成本上限：参数量不超过 3.50M，GFLOPs 不超过 22.00。

## 预注册通过门槛

候选仅在同时满足以下条件时进入 150 轮、三随机种子复验，再开展“仅 P5-lite / P5-lite+空间门控”的组件消融：

1. Large AP 相对 50轮对照提高至少 **+1.00 pp**；
2. Small AP 相对对照不低于 **-0.30 pp**；
3. mAP50-95 相对对照不低于 **-0.10 pp**；
4. 参数量 ≤ 3.50M 且 GFLOPs ≤ 22.00。

任一条件不满足，即停止此精确配置；结果会保留为负向证据，不以单项改善替代联合通过。

## 运行与可追溯性

- 运行脚本：`scripts/run_p5lite_gated_screen.ps1`
- 报告生成：`scripts/generate_p5lite_gated_report.py`
- 训练、标准验证、尺寸验证日志：`logs/`（本地留存，不纳入 Git）
- 结构化指标、比较表与图：`metrics/`、`figures/`
- 输出运行目录：`runs/p5lite_gated_seed0_50e/` 及对应评估目录；脚本会拒绝覆盖已有目录。
