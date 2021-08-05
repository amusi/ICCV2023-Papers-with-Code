# ICCV2021-Papers-with-Code

[ICCV 2021](http://iccv2021.thecvf.com/) 论文和开源项目合集(papers with code)！

1617 papers accepted - 25.9% acceptance rate

ICCV 2021 收录论文IDs：https://docs.google.com/spreadsheets/u/1/d/e/2PACX-1vRfaTmsNweuaA0Gjyu58H_Cx56pGwFhcTYII0u1pg0U7MbhlgY0R6Y-BbK3xFhAiwGZ26u3TAtN5MnS/pubhtml

> 注1：欢迎各位大佬提交issue，分享ICCV 2021论文和开源项目！
>
> 注2：关于往年CV顶会论文以及其他优质CV论文和大盘点，详见： https://github.com/amusi/daily-paper-computer-vision

## 【ICCV 2021 论文和开源目录】

- [Backbone](#Backbone)
- [Transformer](#Transformer)
- [GAN](#GAN)
- [NAS](#NAS)
- [NeRF](#NeRF)
- [Loss](#Loss)
- [Zero-Shot Learning](#Zero-Shot-Learning)
- [Few-Shot Learning](#Few-Shot-Learning)
- [长尾(Long-tailed)](#Long-tailed)
- [Vision and Language](#VL)
- [无监督/自监督(Self-Supervised)](#Un/Self-Supervised)
- [2D目标检测(Object Detection)](#Object-Detection)
- [语义分割(Semantic Segmentation)](#Semantic-Segmentation)
- [实例分割(Instance Segmentation)](#Instance-Segmentation)
- [医学图像分割(Medical Image Segmentation)](#Medical-Image-Segmentation)
- [Few-shot Segmentation](#Few-shot-Segmentation)
- [人体运动分割(Human Motion Segmentation)](#HMS)
- [目标跟踪(Object Tracking)](#Object-Tracking)
- [3D Point Cloud](#3D-Point-Cloud)
- [Point Cloud Object Detection(点云目标检测)](#Point-Cloud-Object-Detection)
- [Point Cloud Semantic Segmenation(点云语义分割)](#Point-Cloud-Semantic-Segmentation)
- [Point Cloud Denoising(点云去噪)](#Point-Cloud-Denoising)
- [Point Cloud Registration(点云配准)](#Point-Cloud-Registration)
- [超分辨率(Super-Resolution)](#Super-Resolution)
- [行人重识别(Person Re-identification)](#Re-ID)
- [2D/3D人体姿态估计(2D/3D Human Pose Estimation)](#Human-Pose-Estimation)
- [3D人头重建(3D Head Reconstruction)](#3D-Head-Reconstruction)
- [行为识别(Action Recognition)](#Action-Recognition)
- [时序动作定位(Temporal Action Localization)](#Temporal-Action-Localization)
- [文本检测(Text Detection)](#Text-Detection)
- [文本识别(Text Recognition)](#Text-Recognition)
- [视觉问答(Visual Question Answering, VQA)](#Visual-Question-Answering)
- [对抗攻击(Adversarial Attack)](#Adversarial-Attack)
- [深度估计(Depth Estimation)](#Depth-Estimation)
- [视线估计(Gaze Estimation)](#Gaze-Estimation)
- [人群计数(Crowd Counting)](#Crowd-Counting)
- [轨迹预测(Trajectory Prediction)](#Trajectory-Prediction)
- [异常检测(Anomaly Detection)](#Anomaly-Detection)
- [场景图生成(Scene Graph Generation)](#Scene-Graph-Generation)
- [Unsupervised Domain Adaptation](#UDA)
- [Video Rescaling](#Video-Rescaling)
- [Hand-Object Interaction](#Hand-Object-Interaction)
- [数据集(Datasets)](#Datasets)
- [其他(Others)](#Others)

<a name="Backbone"></a>

# Backbone

**Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions**

- Paper(Oral): https://arxiv.org/abs/2102.12122
- Code: https://github.com/whai362/PVT

**AutoFormer: Searching Transformers for Visual Recognition**

- Paper: https://arxiv.org/abs/2107.00651
- Code: https://github.com/microsoft/AutoML

**Bias Loss for Mobile Neural Networks**

- Paper: https://arxiv.org/abs/2107.11170
- Code: None

**Vision Transformer with Progressive Sampling**

- Paper: https://arxiv.org/abs/2108.01684

- Code: https://github.com/yuexy/PS-ViT

<a name="Transformer"></a>

# Visual Transformer

**Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**

- Paper: https://arxiv.org/abs/2103.14030
- Code: https://github.com/microsoft/Swin-Transformer

**An Empirical Study of Training Self-Supervised Vision Transformers**

- Paper(Oral): https://arxiv.org/abs/2104.02057
- MoCo v3 Code: None

**Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions**

- Paper(Oral): https://arxiv.org/abs/2102.12122
- Code: https://github.com/whai362/PVT

**Group-Free 3D Object Detection via Transformers**

- Paper: https://arxiv.org/abs/2104.00678
- Code: None

**Spatial-Temporal Transformer for Dynamic Scene Graph Generation**

- Paper: https://arxiv.org/abs/2107.12309
- Code: None

**Rethinking and Improving Relative Position Encoding for Vision Transformer**

- Paper: https://arxiv.org/abs/2107.14222
- Code: https://github.com/microsoft/AutoML/tree/main/iRPE

**Emerging Properties in Self-Supervised Vision Transformers**

- Paper: https://arxiv.org/abs/2104.14294
- Code: https://github.com/facebookresearch/dino

**Learning Spatio-Temporal Transformer for Visual Tracking**

- Paper: https://arxiv.org/abs/2103.17154
- Code: https://github.com/researchmm/Stark

**Fast Convergence of DETR with Spatially Modulated Co-Attention**

- Paper: https://arxiv.org/abs/2101.07448
- Code: https://github.com/abc403/SMCA-replication

**Vision Transformer with Progressive Sampling**

- Paper: https://arxiv.org/abs/2108.01684

- Code: https://github.com/yuexy/PS-ViT

<a name="GAN"></a>

# GAN

**Labels4Free: Unsupervised Segmentation using StyleGAN**

- Homepage: https://rameenabdal.github.io/Labels4Free/
- Paper: https://arxiv.org/abs/2103.14968

**GNeRF: GAN-based Neural Radiance Field without Posed Camera**

- Paper(Oral): https://arxiv.org/abs/2103.15606

- Code: https://github.com/MQ66/gnerf

**EigenGAN: Layer-Wise Eigen-Learning for GANs**

- Paper: https://arxiv.org/abs/2104.12476
- Code: https://github.com/LynnHo/EigenGAN-Tensorflow

**From Continuity to Editability: Inverting GANs with Consecutive Images**

- Paper: https://arxiv.org/abs/2107.13812
- Code: https://github.com/Qingyang-Xu/InvertingGANs_with_ConsecutiveImgs

<a name="NAS"></a>

# NAS

**AutoFormer: Searching Transformers for Visual Recognition**

- Paper: https://arxiv.org/abs/2107.00651

- Code: https://github.com/microsoft/AutoML

<a name="NeRF"></a>

# NeRF

**GNeRF: GAN-based Neural Radiance Field without Posed Camera**

- Paper(Oral): https://arxiv.org/abs/2103.15606

- Code: https://github.com/MQ66/gnerf

**KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs**

- Paper: https://arxiv.org/abs/2103.13744

- Code: https://github.com/creiser/kilonerf

**In-Place Scene Labelling and Understanding with Implicit Scene Representation**

- Homepage: https://shuaifengzhi.com/Semantic-NeRF/
- Paper(Oral): https://arxiv.org/abs/2103.15875

**Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis**

- Homepage: https://ajayj.com/dietnerf
- Paper(DietNeRF): https://arxiv.org/abs/2104.00677

<a name="Loss"></a>

# Loss

**Rank & Sort Loss for Object Detection and Instance Segmentation**

- Paper(Oral): https://arxiv.org/abs/2107.11669
- Code: https://github.com/kemaloksuz/RankSortLoss

**Bias Loss for Mobile Neural Networks**

- Paper: https://arxiv.org/abs/2107.11170
- Code: None

<a name="Zero-Shot-Learning"></a>

# Zero-Shot Learning

**FREE: Feature Refinement for Generalized Zero-Shot Learning**

- Paper: https://arxiv.org/abs/2107.13807
- Code: https://github.com/shiming-chen/FREE

<a name="Few-Shot-Learning"></a>

# Few-Shot Learning

**Few-Shot and Continual Learning with Attentive Independent Mechanisms**

- Paper: https://arxiv.org/abs/2107.14053
- Code: https://github.com/huang50213/AIM-Fewshot-Continual

<a name="Long-tailed"></a>

# 长尾(Long-tailed)

**Parametric Contrastive Learning**

- Paper: https://arxiv.org/abs/2107.12028
- Code: https://github.com/jiequancui/Parametric-Contrastive-Learning

<a name="VL"></a>

# Vision and Language

**VLGrammar: Grounded Grammar Induction of Vision and Language**

- Paper: https://arxiv.org/abs/2103.12975
- Code: https://github.com/evelinehong/VLGrammar

<a name="Un/Self-Supervised"></a>

# 无监督/自监督(Un/Self-Supervised)

**An Empirical Study of Training Self-Supervised Vision Transformers**

- Paper(Oral): https://arxiv.org/abs/2104.02057
- MoCo v3 Code: None

**DetCo: Unsupervised Contrastive Learning for Object Detection**

- Paper: https://arxiv.org/abs/2102.04803

- Code: https://github.com/xieenze/DetCo

<a name="Object-Detection"></a>

# 2D目标检测(Object Detection)

**DetCo: Unsupervised Contrastive Learning for Object Detection**

- Paper: https://arxiv.org/abs/2102.04803
- Code: https://github.com/xieenze/DetCo

**Detecting Invisible People**

- Homepage: http://www.cs.cmu.edu/~tkhurana/invisible.htm
- Code: https://arxiv.org/abs/2012.08419

**Active Learning for Deep Object Detection via Probabilistic Modeling**

- Paper: https://arxiv.org/abs/2103.16130
- Code: None

**Conditional Variational Capsule Network for Open Set Recognition**

- Paper: https://arxiv.org/abs/2104.09159
- Code: https://github.com/guglielmocamporese/cvaecaposr

**MDETR : Modulated Detection for End-to-End Multi-Modal Understanding**

- Homepage: https://ashkamath.github.io/mdetr_page/
- Paper(Oral): https://arxiv.org/abs/2104.12763
- Code: https://github.com/ashkamath/mdetr

**Rank & Sort Loss for Object Detection and Instance Segmentation**

- Paper(Oral): https://arxiv.org/abs/2107.11669
- Code: https://github.com/kemaloksuz/RankSortLoss

**SimROD: A Simple Adaptation Method for Robust Object Detection**

- Paper(Oral): https://arxiv.org/abs/2107.13389
- Code: None

**GraphFPN: Graph Feature Pyramid Network for Object Detection**

- Paper: https://arxiv.org/abs/2108.00580
- Code: None

**Fast Convergence of DETR with Spatially Modulated Co-Attention**

- Paper: https://arxiv.org/abs/2101.07448
- Code: https://github.com/abc403/SMCA-replication

## 半监督目标检测

**End-to-End Semi-Supervised Object Detection with Soft Teacher**

- Paper: https://arxiv.org/abs/2106.09018
- Code: None

<a name="Semantic-Segmentation"></a>

## 语义分割(Semantic Segmentation)

**Personalized Image Semantic Segmentation**

- Paper: https://arxiv.org/abs/2107.13978
- Code: https://github.com/zhangyuygss/PIS
- Dataset: https://github.com/zhangyuygss/PIS

**Standardized Max Logits: A Simple yet Effective Approach for Identifying Unexpected Road Obstacles in Urban-Scene Segmentation**

- Paper(Oral): https://arxiv.org/abs/2107.11264
- Code: None

## 半监督语义分割(Semi-supervised Semantic Segmentation)

**Leveraging Auxiliary Tasks with Affinity Learning for Weakly Supervised Semantic Segmentation**

- Paper: https://arxiv.org/abs/2107.11787
- Code: None

**Re-distributing Biased Pseudo Labels for Semi-supervised Semantic Segmentation: A Baseline Investigation**

- Paper(Oral): https://arxiv.org/abs/2107.11279
- Code: https://github.com/CVMI-Lab/DARS

## 无监督分割(Unsupervised Segmentation)

**Labels4Free: Unsupervised Segmentation using StyleGAN**

- Homepage: https://rameenabdal.github.io/Labels4Free/
- Paper: https://arxiv.org/abs/2103.14968

<a name="Instance-Segmentation"></a>

# 实例分割(Instance Segmentation)

**Instances as Queries**

- Paper: https://arxiv.org/abs/2105.01928
- Code: https://github.com/hustvl/QueryInst

**Crossover Learning for Fast Online Video Instance Segmentation**

- Paper: https://arxiv.org/abs/2104.05970
- Code: https://github.com/hustvl/CrossVIS

**Rank & Sort Loss for Object Detection and Instance Segmentation**

- Paper(Oral): https://arxiv.org/abs/2107.11669
- Code: https://github.com/kemaloksuz/RankSortLoss

<a name="Medical-Image-Segmentation"></a>

# 医学图像分割(Medical Image Segmentation)

**Recurrent Mask Refinement for Few-Shot Medical Image Segmentation**

- Paper: https://arxiv.org/abs/2108.00622
- Code:  https://github.com/uci-cbcl/RP-Net 

<a name="Few-shot-Segmentation"></a>

# Few-shot Segmentation

**Mining Latent Classes for Few-shot Segmentation**

- Paper(Oral): https://arxiv.org/abs/2103.15402
- Code: https://github.com/LiheYoung/MiningFSS

<a name="HMS"></a>

# 人体运动分割(Human Motion Segmentation)

**Graph Constrained Data Representation Learning for Human Motion Segmentation**

- Paper: https://arxiv.org/abs/2107.13362
- Code: None

<a name="Object-Tracking"></a>

# 目标跟踪(Object Tracking)

**Learning Spatio-Temporal Transformer for Visual Tracking**

- Paper: https://arxiv.org/abs/2103.17154
- Code: https://github.com/researchmm/Stark

**Learning to Adversarially Blur Visual Object Tracking**

- Paper: https://arxiv.org/abs/2107.12085
- Code: https://github.com/tsingqguo/ABA

**HiFT: Hierarchical Feature Transformer for Aerial Tracking**

- Paper: https://arxiv.org/abs/2108.00202
- Code: https://github.com/vision4robotics/HiFT

**Learn to Match: Automatic Matching Network Design for Visual Tracking**

- Paper: https://arxiv.org/abs/2108.00803
- Code: https://github.com/JudasDie/SOTS

<a name="3D-Point-Cloud"></a>

# 3D Point Cloud

**Unsupervised Point Cloud Pre-Training via View-Point Occlusion, Completion**

- Homepage: https://hansen7.github.io/OcCo/
- Paper: https://arxiv.org/abs/2010.01089 
- Code: https://github.com/hansen7/OcCo

<a name="Point-Cloud-Object-Detection"></a>

# Point Cloud Object Detection(点云目标检测)

**Group-Free 3D Object Detection via Transformers**

- Paper: https://arxiv.org/abs/2104.00678
- Code: None

<a name="Point-Cloud-Semantic-Segmentation"></a>

## Point Cloud Semantic Segmentation(点云语义分割)

**ReDAL: Region-based and Diversity-aware Active Learning for Point Cloud Semantic Segmentation**

- Paper: https://arxiv.org/abs/2107.11769
- Code: None

**Learning with Noisy Labels for Robust Point Cloud Segmentation**

- Homepage: https://shuquanye.com/PNAL_website/
- Paper(Oral): https://arxiv.org/abs/2107.14230

**VMNet: Voxel-Mesh Network for Geodesic-Aware 3D Semantic Segmentation**

- Paper(Oral): https://arxiv.org/abs/2107.13824
- Code: https://github.com/hzykent/VMNet

<a name="Point-Cloud-Denoising"></a>

## Point Cloud Denoising(点云去噪)

**Score-Based Point Cloud Denoising**

- Paper: https://arxiv.org/abs/2107.10981
- Code: None

<a name="Point-Cloud-Registration"></a>

## Point Cloud Registration(点云配准)

**HRegNet: A Hierarchical Network for Large-scale Outdoor LiDAR Point Cloud Registration**

- Homepage: https://ispc-group.github.io/hregnet
- Paper: https://arxiv.org/abs/2107.11992
- Code: https://github.com/ispc-lab/HRegNet

<a name="Super-Resolution"></a>

# 超分辨率(Super-Resolution)

**Learning for Scale-Arbitrary Super-Resolution from Scale-Specific Networks**

- Paper: https://arxiv.org/abs/2004.03791
- Code: https://github.com/LongguangWang/ArbSR

<a name="Re-ID"></a>

# 行人重识别(Person Re-identification)

**TransReID: Transformer-based Object Re-Identification**

- Paper: https://arxiv.org/abs/2102.04378
- Code: https://github.com/heshuting555/TransReID

<a name="Human-Pose-Estimation"></a>

# 2D/3D人体姿态估计(2D/3D Human Pose Estimation)

## 2D 人体姿态估计

**Human Pose Regression with Residual Log-likelihood Estimation**

- Paper(Oral): https://arxiv.org/abs/2107.11291
- Code(RLE): https://github.com/Jeff-sjtu/res-loglikelihood-regression

## 3D 人体姿态估计

**Probabilistic Monocular 3D Human Pose Estimation with Normalizing Flows**

- Paper: https://arxiv.org/abs/2107.13788
- Code: https://github.com/twehrbein/Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows

<a name="3D-Head-Reconstruction"></a>

# 3D人头重建(3D Head Reconstruction)

**H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction**

- Homepage: https://crisalixsa.github.io/h3d-net/

- Paper: https://arxiv.org/abs/2107.12512

<a name="Action-Recognition"></a>

# 行为识别(Action Recognition)

**MGSampler: An Explainable Sampling Strategy for Video Action Recognition**

- Paper: https://arxiv.org/abs/2104.09952
- Code: None

**Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition**

- Paper: https://arxiv.org/abs/2107.12213
- Code: https://github.com/Uason-Chen/CTR-GCN

<a name="Temporal-Action-Localization"></a>

# 时序动作定位(Temporal Action Localization)

**Enriching Local and Global Contexts for Temporal Action Localization**

- Paper: https://arxiv.org/abs/2107.12960
- Code: None

<a name="Text-Detection"></a>

# 文本检测(Text Detection)

**Adaptive Boundary Proposal Network for Arbitrary Shape Text Detection**

- Paper: https://arxiv.org/abs/2107.12664
- Code: https://github.com/GXYM/TextBPN

<a name="Text-Recognition"></a>

# 文本识别(Text Recognition)

**Joint Visual Semantic Reasoning: Multi-Stage Decoder for Text Recognition**

- Paper: https://arxiv.org/abs/2107.12090
- Code: None

<a name="Visual-Question-Answering"></a>

# 视觉问答(Visual Question Answering, VQA)

**Greedy Gradient Ensemble for Robust Visual Question Answering**

- Paper: https://arxiv.org/abs/2107.12651

- Code: https://github.com/GeraldHan/GGE

<a name="Adversarial-Attack"></a>

# 对抗攻击(Adversarial Attack)

**Feature Importance-aware Transferable Adversarial Attacks**

- Paper: https://arxiv.org/abs/2107.14185
- Code: https://github.com/hcguoO0/FIA

<a name="Depth-Estimation"></a>

# 深度估计(Depth Estimation)

## 单目深度估计

**MonoIndoor: Towards Good Practice of Self-Supervised Monocular Depth Estimation for Indoor Environments**

- Paper: https://arxiv.org/abs/2107.12429
- Code: None

<a name="Gaze-Estimation"></a>

# 视线估计(Gaze Estimation)

**Generalizing Gaze Estimation with Outlier-guided Collaborative Adaptation**

- Paper: https://arxiv.org/abs/2107.13780
- Code: https://github.com/DreamtaleCore/PnP-GA

<a name="Crowd-Counting"></a>

# 人群计数(Crowd Counting)

**Rethinking Counting and Localization in Crowds:A Purely Point-Based Framework**

- Paper(Oral): https://arxiv.org/abs/2107.12746
- Code(P2PNet): https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet

**Uniformity in Heterogeneity:Diving Deep into Count Interval Partition for Crowd Counting**

- Paper: https://arxiv.org/abs/2107.12619
- Code: https://github.com/TencentYoutuResearch/CrowdCounting-UEPNet

<a name="Trajectory-Prediction"></a>

# 轨迹预测(Trajectory Prediction)

**Human Trajectory Prediction via Counterfactual Analysis**

- Paper: https://arxiv.org/abs/2107.14202
- Code: https://github.com/CHENGY12/CausalHTP

**Personalized Trajectory Prediction via Distribution Discrimination**

- Paper: https://arxiv.org/abs/2107.14204
- Code: https://github.com/CHENGY12/DisDis

<a name="Anomaly-Detection"></a>

# 异常检测(Anomaly Detection)

**Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning**

- Paper: https://arxiv.org/abs/2101.10030
- Code: https://github.com/tianyu0207/RTFM

<a name="Scene-Graph-Generation"></a>

# 场景图生成(Scene Graph Generation)

**Spatial-Temporal Transformer for Dynamic Scene Graph Generation**

- Paper: https://arxiv.org/abs/2107.12309
- Code: None

<a name="UDA"></a>

# Unsupervised Domain Adaptation

**Recursively Conditional Gaussian for Ordinal Unsupervised Domain Adaptation**

- Paper(Oral): https://arxiv.org/abs/2107.13467
- Code: None

<a name="Video-Rescaling"></a>

# Video Rescaling

**Self-Conditioned Probabilistic Learning of Video Rescaling**

- Paper: https://arxiv.org/abs/2107.11639

- Code: None

<a name="Hand-Object-Interaction"></a>

# Hand-Object Interaction

**Learning a Contact Potential Field to Model the Hand-Object Interaction**

- Paper: https://arxiv.org/abs/2012.00924
- Code: https://lixiny.github.io/CPF 

<a name="Datasets"></a>

# 数据集(Datasets)

**Personalized Image Semantic Segmentation**

- Paper: https://arxiv.org/abs/2107.13978
- Code: https://github.com/zhangyuygss/PIS
- Dataset: https://github.com/zhangyuygss/PIS

**H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction**

- Homepage: https://crisalixsa.github.io/h3d-net/

- Paper: https://arxiv.org/abs/2107.12512

<a name="Others"></a>

# 其他(Others)

**Progressive Correspondence Pruning by Consensus Learning**

- Homepage: https://sailor-z.github.io/projects/CLNet.html
- Paper: https://arxiv.org/abs/2101.00591
- Code: https://github.com/sailor-z/CLNet

项目主页：

**Energy-Based Open-World Uncertainty Modeling for Confidence Calibration**

- Paper: https://arxiv.org/abs/2107.12628
- Code: None

**Generalized Shuffled Linear Regression**

- Paper: https://drive.google.com/file/d/1Qu21VK5qhCW8WVjiRnnBjehrYVmQrDNh/view?usp=sharing
- Code: https://github.com/SILI1994/Generalized-Shuffled-Linear-Regression

**Discovering 3D Parts from Image Collections**

- Homepage: https://chhankyao.github.io/lpd/

- Paper: https://arxiv.org/abs/2107.13629

**Semi-Supervised Active Learning with Temporal Output Discrepancy**

- Paper: https://arxiv.org/abs/2107.14153
- Code: https://github.com/siyuhuang/TOD

**Why Approximate Matrix Square Root Outperforms Accurate SVD in Global Covariance Pooling?**

Paper: https://arxiv.org/abs/2105.02498

Code: https://github.com/KingJamesSong/DifferentiableSVD 

**Hand-Object Contact Consistency Reasoning for Human Grasps Generation**

- Homepage: https://hwjiang1510.github.io/GraspTTA/
- Paper(Oral): https://arxiv.org/abs/2104.03304
- Code: None

**Equivariant Imaging: Learning Beyond the Range Space**

- Paper(Oral): https://arxiv.org/abs/2103.14756
- Code: https://github.com/edongdongchen/EI

**Just Ask: Learning to Answer Questions from Millions of Narrated Videos**

- Paper(Oral): https://arxiv.org/abs/2012.00451
- Code: https://github.com/antoyang/just-ask
