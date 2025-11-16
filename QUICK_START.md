# CharCut Quick Start Guide 🚀

这个指南帮助您在最短时间内运行完整实验并找到最佳模型。

## 准备工作 (5分钟)

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备字体

将中文字体文件放到 `assets/fonts/` 目录:

```bash
mkdir -p assets/fonts

# 下载或复制字体文件到此目录
# 示例: NotoSansCJK-Regular.otf, SimSun.ttf, SimHei.ttf, KaiTi.ttf
```

**字体来源**:
- [Noto Sans CJK](https://github.com/notofonts/noto-cjk)
- 系统自带字体 (Windows: C:\Windows\Fonts, Mac: /Library/Fonts)

### 3. 准备演示字符集

```bash
# 创建简单的测试字符集
echo "你好世界测试字符" > assets/charset_demo.txt
```

## 方案A: 快速实验 (15-20分钟) ⚡

**推荐用于**: 快速了解系统，初步比较baseline vs advanced

```bash
bash experiments/run_quick_experiment.sh
```

这将:
- ✅ 生成100张测试图像
- ✅ 训练baseline模型 (15 epochs, ~5分钟)
- ✅ 训练advanced模型 (Stage A: 5 epochs + Joint: 10 epochs, ~10分钟)
- ✅ 自动评估和可视化
- ✅ 显示性能对比

**查看结果**:
```bash
# 查看metrics
cat experiments/results/quick_*/baseline/metrics.json

# 查看可视化对比图
ls experiments/results/quick_*/baseline/visualizations/
```

## 方案B: 完整实验套件 (2-4小时) 🎯

**推荐用于**: 找到最佳模型配置，准备生产部署

```bash
# 小数据集 (500张图，推荐)
bash experiments/run_all_experiments.sh small

# 或中等数据集 (2000张图，最终benchmark)
bash experiments/run_all_experiments.sh medium
```

这将测试:
- **6个baseline配置**: 不同学习率、filter数、batch size
- **3个advanced配置**: 不同训练长度和batch size
- 自动评估所有配置
- 生成详细对比报告

**查看结果**:
```bash
# 1. 查看自动生成的报告
cat experiments/results/run_*/REPORT.md

# 2. 生成可视化图表
python experiments/analyze_results.py --run_dir experiments/results/run_*

# 3. 查看图表和分析
ls experiments/results/run_*/analysis/charts/
cat experiments/results/run_*/analysis/summary.md
```

## 方案C: 手动单个实验

### Baseline训练

```bash
# 1. 生成数据
python src/synth/generate_sprites.py \
    --charset assets/charset_demo.txt \
    --fonts assets/fonts \
    --out data/sprites \
    --size 16

python src/synth/compose_paragraphs.py \
    --sprites data/sprites \
    --out data/demo \
    --num_images 200 \
    --canvas 256

# 2. 训练
python src/train_baseline.py \
    --data data/demo \
    --epochs 30 \
    --batch 4 \
    --lr 1e-3 \
    --out runs/my_baseline

# 3. 推理
python src/infer_baseline.py \
    --ckpt runs/my_baseline/best.pt \
    --input data/demo/images \
    --out outputs/predictions \
    --format binary  # 或 rgba, grayscale

# 4. 评估
python src/evaluate.py \
    --pred outputs/predictions \
    --gt data/demo \
    --output outputs/metrics.json

# 5. 可视化
python src/visualize_results.py \
    --images data/demo/images \
    --gt data/demo \
    --pred outputs/predictions \
    --out outputs/visualizations
```

### Advanced训练

```bash
# Stage A
python src/train_stageA.py \
    --data data/demo \
    --epochs 10 \
    --batch 2 \
    --lr 1e-4 \
    --out runs/my_advanced/stageA

# Joint
python src/train_joint.py \
    --data data/demo \
    --epochs 15 \
    --batch 2 \
    --lr 1e-4 \
    --lr_refine 1e-3 \
    --ckpt_stageA runs/my_advanced/stageA/last.pt \
    --out runs/my_advanced/joint

# 推理
python src/postprocess_export.py \
    --ckpt runs/my_advanced/joint/joint_last.pt \
    --input data/demo/images \
    --out outputs/advanced \
    --format binary
```

## 理解结果

### 关键指标

**Instance-level (实例级)**:
- **F1 Score**: 综合性能指标 (越高越好，0-1)
- **Mean IoU**: 分割质量 (越高越好，0-1)
- **Precision**: 预测准确率
- **Recall**: 检测召回率

**Pixel-level (像素级)**:
- **Accuracy**: 像素分类准确率
- **F1**: 像素级综合性能

**Alpha Quality (Alpha质量)**:
- **MAE**: 平均绝对误差 (越小越好)
- **Gradient Error**: 边缘质量 (越小越好)

### 选择最佳模型

自动报告会按F1分数排名所有模型:

```bash
cat experiments/results/run_*/REPORT.md
```

查找 "Best Model Selection" 部分。

### 可视化解读

可视化图片显示6个面板:
1. **Original**: 原图
2. **GT Overlay**: 真值叠加 (绿色)
3. **Pred Overlay**: 预测叠加 (红色)
4. **GT Mask**: 真值mask
5. **Pred Mask**: 预测mask
6. **Diff**: 差异图
   - 白色 = 正确 (True Positive)
   - 红色 = 误检 (False Positive)
   - 蓝色 = 漏检 (False Negative)

## 常见问题

### Q: 内存不足 (OOM)
```bash
# 减小batch size
--batch 2  # 或 1

# 使用更小数据集
bash experiments/run_all_experiments.sh tiny
```

### Q: 训练太慢
```bash
# 检查是否使用GPU
python -c "import torch; print(torch.cuda.is_available())"

# 减少epochs
--epochs 10

# 增加num_workers
--num_workers 4
```

### Q: 字体加载失败
```bash
# 确认字体文件格式
ls -lh assets/fonts/
# 应该看到 .ttf 或 .otf 文件

# 手动测试字体
python src/synth/generate_sprites.py \
    --charset assets/charset_demo.txt \
    --fonts assets/fonts \
    --out test_sprites \
    --size 16
```

### Q: 结果不好
- 检查数据质量: `ls data/demo/images/`
- 增加训练轮数: `--epochs 50`
- 调整学习率: 尝试 `--lr 5e-4` 或 `--lr 2e-3`
- 增加模型容量: `--base_filters 48`

## 下一步

找到最佳模型后:

### 1. 保存模型
```bash
cp experiments/results/run_*/baseline_default/checkpoints/best.pt models/production.pt
```

### 2. 在新数据上推理
```bash
python src/infer_baseline.py \
    --ckpt models/production.pt \
    --input your_images/ \
    --out your_output/ \
    --format binary
```

### 3. 在真实数据上微调
```bash
# 准备真实数据 (COCO格式)
# data/real/
#   ├── images/
#   ├── masks/
#   └── annotations.json

# 从最佳模型继续训练
python src/train_baseline.py \
    --data data/real \
    --epochs 20 \
    --batch 4 \
    --lr 1e-4 \  # 微调用更小学习率
    --out runs/finetuned

# 加载预训练权重 (需要修改代码支持)
```

### 4. 部署优化
```bash
# 导出ONNX (需要实现)
# 使用TensorRT优化 (需要实现)
# 集成到应用
```

## 实验建议

### 快速迭代
1. 先跑 `run_quick_experiment.sh` 了解baseline
2. 调整一个参数 (如学习率)
3. 手动训练测试
4. 确定好配置后，跑完整实验

### 系统性评估
1. 跑 `run_all_experiments.sh small`
2. 分析报告找到最佳配置
3. 用medium数据集验证
4. 在真实数据上测试

### 生产部署
1. 完整实验找到最佳配置
2. 在大数据集上充分训练
3. 在测试集上评估
4. 优化推理速度
5. 部署监控

## 时间估算 (GPU)

| 任务 | 时间 |
|------|------|
| 快速实验 (100图) | 15-20分钟 |
| 小数据集实验 (500图) | 2-3小时 |
| 中等数据集实验 (2000图) | 4-6小时 |
| 单个baseline训练 (30 epochs, 500图) | 15-20分钟 |
| 单个advanced训练 (10+15 epochs, 500图) | 30-40分钟 |

## 获取帮助

- 实验框架文档: `experiments/README.md`
- 主项目文档: `README.md`
- 问题追踪: 查看日志文件 `*.log`

---

**快速开始命令**:

```bash
# 1分钟快速测试 - 确保环境正常
python src/models/baseline_unet.py

# 15分钟快速实验 - 了解系统
bash experiments/run_quick_experiment.sh

# 3小时完整实验 - 找到最佳模型
bash experiments/run_all_experiments.sh small
```

祝实验顺利! 🎉
