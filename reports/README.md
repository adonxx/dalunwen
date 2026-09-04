# 实验报告索引

本目录保存可直接阅读和复核的实验方案、结论、汇总指标和图表。训练权重、`runs/` 目录、原始控制台日志、逐图预测结果和临时状态文件不纳入 Git；它们保留在本地，可由对应运行脚本和检查点复现。

## 主结论

- [三随机种子核心复现](core_multiseed/README.md)：完整 Drone-YOLO 相比 YOLOv11s 的总体、小目标和轻量化结果。
- [统一消融评估](unified_evaluation/README.md)：MF-FPN、LSCD 和 Inner-WIoU 的单阶段消融。
- [三种子端到端部署基准](deployment_benchmark/README.md)：FP32/FP16 的端到端延迟、FPS 和显存。

## 已完成的筛选实验

- [尺度检测层消融](scale_ablation/README.md)
- [自适应加权融合三种子复验](weighted_fusion_multiseed/README.md)
- [NWD 回归损失](nwd_loss_ablation/README.md)
- [小目标感知 TAL](small_aware_tal/REPORT.md)
- [小目标裁剪训练](small_object_crop/REPORT.md)
- [重叠切片训练](overlap_tile_training/REPORT.md)
- [尺度选择性知识蒸馏](selective_kd/README.md)

## 进行中的结构筛选

- [P5-lite 语义反馈与空间门控（50轮结果）](p5lite_gated_fusion/README.md)
- [P5-lite 空间门控（150轮探索性复验方案）](p5lite_gated_fusion/LONGRUN_PLAN.md)

## 推理策略探索

- [SAHI 切片推理](sahi_inference/README.md)
- [SAHI 尺度融合](sahi_scale_fusion/README.md)
- [Scale48 联合推理](scale48_joint_inference/README.md)
- [密度门控 SAHI](density_gated_sahi/README.md)

## 参考方法笔记

- [小目标检测方法与适配性整理](small_object_detection_methods.md)
- [阶段性实验总结](stage_summary_2026-08-27.md)
