# 尺度梯度冲突诊断方案

## 研究问题

P5-lite仅增强最终P4后，50轮 Large AP 提高1.77 pp，但 Small AP下降0.57 pp、mAP50-95下降0.44 pp。由于P2/P3不再直接接收P5-lite特征，剩余的主要耦合路径包括LSCD共享卷积、共享分类/回归预测器，以及由三个尺度共同更新的主干和颈部。本实验检验：**小目标与大目标监督是否在这些共享参数上形成方向相反的梯度，以及P4-only候选是否放大该冲突。**

## 固定对象与样本

- 对照：`runs/loss_inner_wiou_50e/weights/best.pt`，stage=`full`。
- 候选：`runs/p5lite_p4only_seed0_50e/weights/best.pt`，stage=`full_p5lite_p4only`。
- 两者均为seed=0、50轮、相同数据和初始化来源。
- 从VisDrone2019训练集按固定顺序读取16个batch，batch=4、imgsz=640、关闭数据增强和随机打乱。
- 目标按照网络实际输入坐标分组：small < 32²，medium为32²–96²，large ≥ 96²像素。
- 只选同时包含三种尺度目标的batch；两套模型使用完全相同的图像和标注。

## 梯度定义

对同一批图像分别只保留small、medium或large标注，执行独立前向与反向，记录以下参数组的梯度：

1. `shared_features`：LSCD两层共享3×3 Conv-GN；
2. `shared_predictors`：LSCD共享回归与分类1×1预测器；
3. `deep_backbone`：P4深层主干（模型层5–8）；
4. `neck`：MF-FPN颈部参数。

主分析使用box + DFL定位损失，减少“移除其他尺度标注后被视为背景”对分类项的干扰；总检测损失作为敏感性分析。每个batch计算small–medium、small–large、medium–large的梯度余弦相似度：正值表示方向一致，接近0表示近似正交，负值表示存在冲突。

## 预注册判断

若P4-only候选在`shared_features`或`shared_predictors`上满足以下全部条件，则认为有足够证据进入“P4检测头解耦”50轮实验：

- small–large定位梯度平均余弦相似度 < -0.05；
- small–large定位梯度为负的batch比例 ≥ 60%；
- 结果至少基于16个有效batch。

若只在总损失上出现负值、定位损失不冲突，则只能说明分类/背景项可能冲突，不能直接归因于LSCD特征共享；应先调整诊断或损失隔离方式。若共享头没有达到上述条件，则不实施P4头解耦，停止当前P5修复路线。

## 输出

- 原始逐batch结果：`metrics/gradient_cosines.csv`
- 汇总统计与判定：`metrics/gradient_summary.csv`、`metrics/decision.json`
- 样本清单：`metrics/sample_manifest.json`
- 可视化：`figures/`
- 简明结论：`README.md`
