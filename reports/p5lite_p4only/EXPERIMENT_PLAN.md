# P5-lite 仅增强 P4：分阶段实验方案

## 研究问题

P5-lite 全尺度反馈在50轮时暂时提高 Large AP，但150轮后 mAP50-95、Small AP、Medium AP 和 Large AP 均低于 Full 对照。由于该模块位于 top-down 路径之前，P5-lite 信号会同时改变 P4、P3 和 P2。本实验通过只改变模块位置，检验失败是否来自过宽的跨尺度传播范围。

## 单因素结构

```text
原 MF-FPN：P4 → P3 → P2 → P3 → 最终P4
                │      │          │
                └P2检测└P3检测    └P5-lite生成/上采样/空间门控 → P4检测
```

- P2、P3 的结构、索引和直接输入完全保持 Full 模型不变。
- P5-lite 从最终 P4 产生20×20语义特征，上采样后只残差增强 P4检测输入。
- 保持128隐藏通道、gate bias=-2、LSCD及Inner-WIoU不变。
- 该设计只隔离直接特征传播；LSCD共享卷积仍可能通过联合训练产生间接尺度耦合。

## 第一阶段：50轮筛选

- 数据：VisDrone2019，imgsz=640，batch=4，SGD，seed=0，50 epochs。
- 初始化：从 `yolo11s.pt` 独立训练。
- 对照：`loss_inner_wiou_50e`，mAP50-95 20.63%、Small AP 12.93%、Medium AP 30.38%、Large AP 35.20%。
- 固定验证：conf=0.001、IoU=0.7、max_det=300、FP32及COCO原图面积分桶。

同时满足下列条件才继续150轮：

1. Large AP Δ ≥ +1.00 pp；
2. Small AP Δ ≥ -0.30 pp；
3. mAP50-95 Δ ≥ -0.10 pp；
4. 参数量 ≤ 3.50M，GFLOPs ≤ 22.00。

## 第二阶段：条件式150轮复验

仅在50轮全部通过时，从 `yolo11s.pt` 重新独立训练150轮，不续训50轮权重。对照为 `full_paper` 的 seed=0、150轮结果。150轮沿用同一判定门槛；通过后才建议补 seed=1、2，未通过则停止P5-lite路线。

## 自动记录

- 50轮：`SCREEN_REPORT.md`、`metrics/*50e*`、`figures/*50e*`
- 150轮（条件触发）：`LONGRUN_REPORT.md`、`metrics/*150e*`、`figures/*150e*`
- 日志和状态保存在 `logs/` 与状态JSON中；运行脚本拒绝覆盖已有实验目录。
