# P5-lite 空间门控：150轮探索性复验方案

## 目的与定位

50轮筛选中，P5-lite + 空间门控相对 Full 对照的 Large AP 提高 1.74 pp，Small AP 仅下降 0.20 pp，但 mAP50-95 下降 0.20 pp，因超过预注册容忍值而未通过。本实验不修改该结论，也不追认50轮筛选通过；它将候选视为“临界配置”，检验更完整训练预算是否能够保留大目标收益并消除总体精度差距。

## 公平训练与对照

- 候选：`full_p5lite_gate`，结构、128通道 P5-lite 和初始 gate bias=-2 均保持不变。
- 训练：VisDrone2019、640×640、SGD、batch=4、seed=0、150 epochs。
- 初始化：从 `yolo11s.pt` 独立初始化，**不续训50轮检查点**。
- 对照：同为 seed=0、从 `yolo11s.pt` 初始化并训练150轮的 `runs/full_paper/weights/best.pt`。
- 验证：统一 `conf=0.001`、IoU=0.7、max_det=300、FP32，并使用原图像素 COCO 面积分桶。
- 对照固定结果：mAP50 41.40%、mAP50-95 24.34%、Small AP 15.62%、Medium AP 35.24%、Large AP 46.04%。

## 预注册判定

150轮候选只有同时满足以下条件，才值得继续 seed=1、2：

1. Large AP 相对 Full 150轮对照提高至少 **+1.00 pp**；
2. Small AP 相对对照下降不超过 **-0.30 pp**；
3. mAP50-95 相对对照下降不超过 **-0.10 pp**；
4. 参数量 ≤ 3.50M，GFLOPs ≤ 22.00。

若未同时满足，则终止当前“全尺度反馈”配置，下一项结构实验改为 P5-lite 仅增强最终 P4 分支，不让其进入 P3/P2。

## 自动产物

- 训练：`runs/p5lite_gated_seed0_150e/`
- 标准与尺寸验证：对应的 `_eval`、`_size_eval` 运行目录
- 日志：`reports/p5lite_gated_fusion/logs/`
- 指标与比较表：`reports/p5lite_gated_fusion/metrics/`
- 图表及最终结论：`figures/`、`LONGRUN_REPORT.md`
- 运行脚本会拒绝覆盖任何已有目录。
