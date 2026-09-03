# 小目标检测代表性方法与 Drone-YOLO 可迁移方案

> 整理时间：2026-08-23  
> 范围：不限于无人机领域，覆盖通用目标检测、交通监控、遥感、红外和高分辨率图像。本文仅纳入能够找到作者论文页、会议论文页或正式论文 PDF 的代表性工作。

## 1. 整理结论

小目标检测的困难可以归纳为四类：输入缩放后有效像素过少；浅层特征有空间细节但语义不足；IoU 对微小框的位置偏差过度敏感；目标在高分辨率图像中稀疏分布，直接提高分辨率会显著增加计算量。已有研究主要沿六条路线展开：高分辨率特征与多尺度融合、图像切片与区域放大、上下文建模、微小目标特征增强、尺度感知标签分配与回归、知识蒸馏及稀疏计算。

当前 Drone-YOLO 已经通过 P2 检测层和 MF-FPN 覆盖了“保留高分辨率特征”路线。新完成的尺寸评估显示，完整模型相对基线的 Small AP 提升 2.09 个百分点、Small AP50 提升 3.32 个百分点，但 Large AP 下降 2.12 个百分点。因此，下一阶段不宜继续无条件增加高分辨率分支，而应优先尝试不显著增加推理复杂度的尺度感知监督、NWD/RFLA 类标签分配，以及只在推理阶段启用的切片对照。

## 2. 代表性论文及具体做法

### 2.1 高分辨率特征与多尺度融合

| 方法 | 核心问题 | 具体做法 | 优点与代价 | 对当前项目的启示 |
|---|---|---|---|---|
| FPN（CVPR 2017）[1] | 浅层分辨率高但语义弱 | 建立自顶向下路径和横向连接，将高层语义传递到高分辨率特征层，并在多个尺度上分别预测 | 通用、稳定；额外计算较小，但简单相加可能忽略不同尺度贡献差异 | 当前 MF-FPN 属于该路线，可作为理论基础；无需再单独复现普通 FPN |
| EfficientDet / BiFPN（CVPR 2020）[7] | 不同尺度输入对融合结果的贡献不相等 | 删除无融合价值的单输入节点，重复双向特征路径，并用可学习归一化权重融合不同尺度 | 精度—效率平衡较好；重复 BiFPN 仍会增加延迟 | 可在 MF-FPN 融合节点增加轻量可学习权重，做“等权融合 vs 加权融合”小消融 |
| Deformable DETR（ICLR 2021）[8] | 全局注意力计算昂贵，普通 DETR 对小目标和高分辨率特征不友好 | 每个查询只在参考点附近采样少量多尺度位置，以可变形注意力聚合信息 | 小目标和收敛速度明显优于原始 DETR；完整替换 YOLO 检测头工程量较大 | 更适合作为注意力设计参考，可尝试仅在 P2/P3 融合处加入轻量可变形卷积，而非整体改成 DETR |
| QueryDet（CVPR 2022）[12] | P2 等高分辨率检测头的大量计算浪费在背景 | 先在低分辨率层预测小目标粗位置，再用级联稀疏查询只计算高分辨率特征中的候选区域 | 同时利用高分辨率并减少冗余；依赖稀疏卷积和定制推理 | 与当前 P2 检测层高度相关，但实现风险较高，适合作为后续“轻量化与小目标兼顾”的深化方向 |

### 2.2 尺度归一化、切片与区域放大

| 方法 | 核心问题 | 具体做法 | 优点与代价 | 对当前项目的启示 |
|---|---|---|---|---|
| SNIP（CVPR 2018）[3] | 同一网络难以同时学习极小和极大实例 | 在图像金字塔的不同输入尺度上，只对落入指定尺寸区间的目标反向传播 | 缓解尺度分布冲突；多尺度训练和推理成本较高 | 可改造成 YOLO 的尺度感知采样或损失掩码，优先解决当前 Small AP 上升、Large AP 下降的权衡 |
| SNIPER（NeurIPS 2018）[4] | 完整图像金字塔浪费大量背景计算 | 围绕合适尺度的目标生成固定尺寸 chips，并通过 RPN 挖掘困难负样本 | 保留多尺度收益同时减少像素处理量；数据预处理较复杂 | 可在训练集生成“含小目标的正切片+困难背景切片”，无需更改网络结构 |
| ClusDet（ICCV 2019）[5] | 高分辨率图像中的目标分布稀疏且常成簇出现 | 用 CPNet生成目标簇区域、ScaleNet估计局部尺度，再把尺度归一化的簇区域送入专用检测器 | 比均匀切片更省计算，并利用局部上下文；需要额外区域网络 | VisDrone 目标密集区域明显，可借鉴其“按密度选择切片”思想，而不是固定网格全图推理 |
| AutoFocus（ICCV 2019）[6] | 多尺度推理对整幅图逐层处理过慢 | 在粗尺度预测可能包含小目标的 FocusPixels，合并为 FocusChips，仅对这些区域进行细尺度检测 | 可接近完整多尺度精度并显著减少处理像素；合并多尺度预测较复杂 | 可作为自适应 SAHI 的设计参考，先做低分辨率定位，再对候选区域放大检测 |
| SAHI（ICIP 2022）[10] | 整图缩放使远距离目标几乎失去像素信息 | 将原图切成有重叠的局部块，分别检测后把坐标映射回原图，并用 NMS/NMM 合并重复框；也可用切片数据微调 | 与检测器解耦、无需改模型即可测试；推理延迟会按切片数增长 | **最高优先级推理对照**：对 baseline/full 采用相同切片尺寸和重叠率，报告 Small AP 与端到端延迟 |

### 2.3 上下文与小目标表征增强

| 方法 | 核心问题 | 具体做法 | 优点与代价 | 对当前项目的启示 |
|---|---|---|---|---|
| Inside-Outside Net（CVPR 2016）[18] | 小目标自身纹理不足，单个 RoI 难以识别 | 用 skip pooling 汇聚多层特征，并用空间循环网络编码目标外部上下文 | 证明上下文和多尺度信息对小目标有效；结构偏两阶段且较重 | 可将“局部目标+周边上下文”转化为 P2 特征上的大感受野轻量卷积或上下文注意力 |
| Perceptual GAN（CVPR 2017）[2] | 小目标特征低分辨率、噪声大，与大目标特征差距明显 | 生成器把小目标特征提升为类似大目标的“超分辨”表征，判别器同时约束其真实性和检测可用性 | 能直接强化弱特征；对抗训练不稳定、训练成本较高 | 可借鉴“大小目标特征对齐”，但优先使用蒸馏/特征模仿替代 GAN |
| CFINet（ICCV 2023）[15] | 小目标正样本候选不足且 RoI 表征不充分 | 通过动态锚框选择和级联回归生成由粗到细的 proposals，再用特征模仿分支和监督对比损失增强小目标 RoI | 对小目标针对性强；建立在 Faster R-CNN 两阶段框架上 | 可迁移“困难小目标向高质量样本特征靠拢”的损失思想，不建议直接移植完整 proposal 流程 |
| SET（CVPR 2025）[17] | 编码过程中微小目标的频率特征被背景高频噪声淹没 | HBS 模块自适应平滑背景高频噪声，API 模块在训练时注入对抗扰动以提升关键区域显著性 | 训练时增强、推理可不增加 API 负担；模块较新，需要充分复验 | 可先分析 VisDrone 小目标区域的频谱，再决定是否在 P2 加轻量背景平滑；不应直接堆叠未经验证的模块 |

### 2.4 标签分配与边界框回归

| 方法 | 核心问题 | 具体做法 | 优点与代价 | 对当前项目的启示 |
|---|---|---|---|---|
| NWD（2021/2022）[9] | 微小框偏移一两个像素就会造成 IoU 剧烈变化 | 将边界框建模为二维高斯分布，用归一化 Wasserstein 距离度量框相似性，可用于标签分配、回归或 NMS | 对极小框位置偏差更平滑；常数和组合权重需要在数据集上验证 | **最高优先级训练消融**：与当前 Inner-WIoU 对照，先测试 `IoU loss`、`NWD loss`、`IoU+NWD` 三组 |
| RFLA（ECCV 2022）[11] | 锚框/点先验与微小目标实际感受野不匹配，正样本分配偏向大目标 | 用高斯感受野距离衡量特征点与目标的匹配程度，再通过分层标签分配平衡不同尺度监督 | 同时面向 anchor-based 和 anchor-free 检测器；需要修改 assigner | 当前 YOLO 属于无锚点检测，可在任务对齐分配器旁增加 RFD 尺度项，重点观察 Small AP 和正样本数量 |
| DCFL（CVPR 2023）[13] | 旋转微小目标中先验、正样本特征和实例之间存在动态错配 | 动态建模先验和目标表征，先粗匹配候选，再用预测质量进行精细约束 | 标签分配更灵活；主要面向旋转框，直接移植成本高 | 可借鉴“粗筛选+质量精排”而非旋转框细节，用于改造小目标正样本选择 |
| SLS Loss / MSHNet（CVPR 2024）[16] | 普通 IoU/Dice 对不同尺度和中心位置的敏感性不足 | 根据目标尺度对 IoU 项加权，并加入中心位置惩罚；在多尺度输出上共同监督 | 模型结构简单，损失可迁移；原任务是红外小目标分割而非矩形框检测 | 可把尺度权重和中心惩罚思想移植到框回归，但必须作为跨任务启发单独验证，不能直接引用其分割结果证明检测有效 |

### 2.5 面向轻量模型的知识蒸馏

| 方法 | 核心问题 | 具体做法 | 优点与代价 | 对当前项目的启示 |
|---|---|---|---|---|
| ScaleKD（CVPR 2023）[14] | 小模型难以兼顾推理速度和小目标表征 | 将教师特征解耦为多尺度嵌入，使学生显式模仿小目标特征；再用跨尺度助手修正学生噪声框 | 与轻量化目标高度一致，推理时可移除教师；训练需要一个明显更强的教师模型 | **高优先级深化方案**：用更强 YOLO11m/l 教师蒸馏 2.77M 完整模型，不能使用精度低于学生的当前基线作教师 |

## 3. 推荐的实验顺序

### 第一阶段：低风险、可快速形成论文证据

1. **SAHI 推理对照**：固定 baseline/full，测试 640×640 与 960×960 切片、20% 重叠；报告总体 AP、Small AP、端到端延迟和重复框比例。
2. **NWD 回归消融**：保持结构、数据增强和随机种子不变，对比 Inner-WIoU、NWD、Inner-WIoU+NWD；先训练 30–50 轮筛选，再完整训练。
3. **尺度感知损失权重**：对 small/medium/large 分别统计训练正样本与回归损失，尝试只提高小目标回归权重，检查是否进一步损害 Large AP。

### 第二阶段：兼顾小目标与轻量化的论文主线

1. **加权 MF-FPN**：把固定融合改为类似 BiFPN 的归一化可学习权重，只保留一层轻量融合，比较参数量、GFLOPs和Small AP。
2. **尺度感知蒸馏**：训练强教师，对 P2/P3/P4 分别蒸馏，并提高 P2 中目标区域的权重；推理端仍使用当前轻量学生。
3. **密度自适应切片**：若 SAHI 精度提升明显但过慢，再借鉴 ClusDet、AutoFocus 或 QueryDet，只放大包含小目标/目标簇的候选区域。

### 第三阶段：高风险探索

1. 频域背景平滑与 SET 式训练扰动。
2. 稀疏查询 P2 检测头。
3. 两阶段 proposal/特征模仿结构。

这些方向应在前两阶段证据充分后再开展，否则容易使硕士论文变成缺少清晰因果关系的模块堆叠。

## 4. 建议的消融表结构

| 实验 | MF-FPN | LSCD | 回归/分配方法 | 切片 | 蒸馏 | Small AP | Medium AP | Large AP | mAP50-95 | 参数量 | 端到端延迟 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline |  |  | Inner-WIoU |  |  | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Full | ✓ | ✓ | Inner-WIoU |  |  | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Full+SAHI | ✓ | ✓ | Inner-WIoU | ✓ |  | 待填 | 待填 | 待填 | 待填 | 不变 | 待填 |
| Full+NWD | ✓ | ✓ | NWD/混合 |  |  | 待填 | 待填 | 待填 | 待填 | 基本不变 | 待填 |
| Full+ScaleKD | ✓ | ✓ | 最优方案 |  | ✓ | 待填 | 待填 | 待填 | 待填 | 推理不变 | 待填 |

## 5. 主张—证据边界

| 主张 | 当前证据 | 状态 |
|---|---|---|
| 完整模型改善 VisDrone 小目标 | Small AP +2.09 pp，Small AP50 +3.32 pp | 已支持，仍需多随机种子 |
| P2/MF-FPN 是小目标收益主要来源 | MF-FPN 阶段 Small AP +2.11 pp | 已支持，单随机种子 |
| 当前结构对所有尺度均有利 | Large AP -2.12 pp | 不支持 |
| SAHI、NWD或ScaleKD会改善当前模型 | 仅有其他论文证据，尚未在当前工程验证 | 需要新实验 |
| 某一跨领域方法可直接证明无人机数据有效 | 数据分布和任务形式不同 | 不成立，必须本地复验 |

## 6. 参考文献与原始入口

[1] Lin T Y, et al. [Feature Pyramid Networks for Object Detection](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf). CVPR, 2017.  
[2] Li J, et al. [Perceptual Generative Adversarial Networks for Small Object Detection](https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Perceptual_Generative_Adversarial_CVPR_2017_paper.html). CVPR, 2017.  
[3] Singh B, Davis L S. [An Analysis of Scale Invariance in Object Detection—SNIP](https://openaccess.thecvf.com/content_cvpr_2018/html/Singh_An_Analysis_of_CVPR_2018_paper.html). CVPR, 2018.  
[4] Singh B, Najibi M, Davis L S. [SNIPER: Efficient Multi-Scale Training](https://proceedings.neurips.cc/paper/2018/hash/166cee72e93a992007a89b39eb29628b-Abstract.html). NeurIPS, 2018.  
[5] Yang F, et al. [Clustered Object Detection in Aerial Images](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_Clustered_Object_Detection_in_Aerial_Images_ICCV_2019_paper.html). ICCV, 2019.  
[6] Najibi M, Singh B, Davis L S. [AutoFocus: Efficient Multi-Scale Inference](https://openaccess.thecvf.com/content_ICCV_2019/html/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.html). ICCV, 2019.  
[7] Tan M, Pang R, Le Q V. [EfficientDet: Scalable and Efficient Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html). CVPR, 2020.  
[8] Zhu X, et al. [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159). ICLR, 2021.  
[9] Wang J, et al. [A Normalized Gaussian Wasserstein Distance for Tiny Object Detection](https://arxiv.org/abs/2110.13389). 2021/2022.  
[10] Akyon F C, Altinuc S O, Temizel A. [Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://arxiv.org/abs/2202.06934). ICIP, 2022.  
[11] Xu C, et al. [RFLA: Gaussian Receptive Field Based Label Assignment for Tiny Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690518.pdf). ECCV, 2022.  
[12] Yang C, Huang Z, Wang N. [QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html). CVPR, 2022.  
[13] Xu C, et al. [Dynamic Coarse-To-Fine Learning for Oriented Tiny Object Detection](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Dynamic_Coarse-To-Fine_Learning_for_Oriented_Tiny_Object_Detection_CVPR_2023_paper.html). CVPR, 2023.  
[14] Zhu Y, et al. [ScaleKD: Distilling Scale-Aware Knowledge in Small Object Detector](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.pdf). CVPR, 2023.  
[15] Yuan X, et al. [Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning](https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_Small_Object_Detection_via_Coarse-to-fine_Proposal_Generation_and_Imitation_Learning_ICCV_2023_paper.html). ICCV, 2023.  
[16] Liu Q, et al. [Infrared Small Target Detection with Scale and Location Sensitivity](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html). CVPR, 2024.  
[17] Sun H, et al. [SET: Spectral Enhancement for Tiny Object Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html). CVPR, 2025.  
[18] Bell S, et al. [Inside-Outside Net: Detecting Objects in Context With Skip Pooling and Recurrent Neural Networks](https://openaccess.thecvf.com/content_cvpr_2016/html/Bell_Inside-Outside_Net_Detecting_CVPR_2016_paper.html). CVPR, 2016.

## 7. 审稿人式自检

| 维度 | 判断 | 后续动作 |
|---|---|---|
| 贡献定位 | 需要进一步聚焦 | 从“NWD损失、尺度蒸馏、密度切片”中选择一条主线，不把所有方法同时加入 |
| 写作清晰度 | 通过 | 保持 Small、tiny、弱小目标等术语定义一致 |
| 实验强度 | 需要新实验 | 所有迁移方法至少与 baseline/full 在同一协议下比较 |
| 评估完整性 | 需要修订 | 增加多随机种子、端到端延迟、峰值显存和失败案例 |
| 方法可靠性 | 需要新实验 | 检查小目标收益是否以大目标性能或推理速度为代价 |

本文中的“推荐优先级”是结合当前复现实验结果作出的研究判断，不代表相应论文已经在本工程中复现；未经本地实验的方法均不得写成已验证贡献。
