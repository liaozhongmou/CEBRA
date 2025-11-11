# CEBRA 合成数据实验脚本 - 完整总结

## 📦 已创建的文件

```
experiments/
├── run_synthetic_experiment.py  # 完整实验脚本（主要文件）
├── quick_test.py               # 快速测试脚本
├── README.md                   # 使用说明
├── INSTALL_GUIDE.md            # 安装指南
├── GIT_WORKFLOW.md             # Git工作流程
└── SUMMARY.md                  # 本文件（总结）
```

## 🎯 脚本功能

### 1. `run_synthetic_experiment.py` - 完整实验流程

**功能**：
- ✅ 加载CEBRA内置合成数据集（5种噪声类型）
- ✅ 训练CEBRA模型（可自定义所有超参数）
- ✅ 评估嵌入质量（使用Ridge回归解码）
- ✅ 生成可视化（2D/3D嵌入图、时间序列图）
- ✅ 保存所有结果（模型、嵌入、指标、图表）

**支持的数据集**：
- `continuous-label-poisson` (默认)
- `continuous-label-gaussian`
- `continuous-label-laplace`
- `continuous-label-uniform`
- `continuous-label-t`

**可配置参数**：
- 模型架构、输出维度、迭代次数
- 批次大小、学习率、温度参数
- 时间偏移、距离度量、条件分布
- 输出目录、训练设备（CPU/GPU）

### 2. `quick_test.py` - 快速验证

**功能**：
- 运行1000次迭代的快速测试
- 生成基础可视化
- 计算解码性能（R²分数）
- 验证环境配置是否正确

**用途**：
- 首次运行，验证安装
- 快速原型测试
- 调试脚本

## 🚀 快速开始

### 步骤1：安装环境（必须先完成）

由于Python版本兼容性问题，需要先设置正确的环境：

```bash
# 选项A：创建新环境（推荐）
conda create -n cebra_exp python=3.9
conda activate cebra_exp
pip install torch torchvision torchaudio
pip install cebra matplotlib seaborn scikit-learn joblib

# 选项B：或安装开发版本
cd /root/cebra_projects/CEBRA-1
pip install -e .
```

详见 `INSTALL_GUIDE.md`

### 步骤2：运行快速测试

```bash
cd /root/cebra_projects/CEBRA-1
python experiments/quick_test.py
```

预期输出：
- 下载合成数据
- 训练1000次迭代（~5分钟）
- 生成 `quick_test_result.png`
- 显示R²分数

### 步骤3：运行完整实验

```bash
python experiments/run_synthetic_experiment.py \
    --dataset continuous-label-poisson \
    --max-iterations 5000 \
    --output-dir ./results/experiment_001
```

## 📊 输出结果

运行完整实验后会生成：

```
results/experiment_001/
├── models/
│   └── cebra_model.pt              # PyTorch模型文件
├── embeddings/
│   └── embedding.npy                # NumPy数组 (n_samples x n_dims)
├── figures/
│   └── embedding_visualization.png  # 4个子图的可视化
└── logs/
    ├── experiment_results.jl        # 配置+指标（joblib）
    └── summary.txt                  # 文本摘要
```

## 🔬 实验示例

### 示例1：比较不同噪声类型

```bash
for noise in poisson gaussian laplace uniform t; do
    python experiments/run_synthetic_experiment.py \
        --dataset continuous-label-${noise} \
        --max-iterations 3000 \
        --output-dir ./results/noise_comparison/${noise}
done
```

### 示例2：超参数搜索（嵌入维度）

```bash
for dim in 2 3 5 8 10; do
    python experiments/run_synthetic_experiment.py \
        --output-dimension ${dim} \
        --max-iterations 5000 \
        --output-dir ./results/dimension/${dim}
done
```

### 示例3：时间偏移实验

```bash
for offset in 1 5 10 20 50; do
    python experiments/run_synthetic_experiment.py \
        --time-offsets ${offset} \
        --max-iterations 5000 \
        --output-dir ./results/timeoffset/${offset}
done
```

## 📈 结果分析

### 加载和查看结果

```python
import joblib
import numpy as np
from cebra import CEBRA
import matplotlib.pyplot as plt

# 加载实验结果
results = joblib.load("results/experiment_001/logs/experiment_results.jl")

print("配置:", results['config'])
print("指标:", results['metrics'])
print("  - 训练 R²:", results['metrics']['train_r2'])
print("  - 测试 R²:", results['metrics']['test_r2'])

# 加载嵌入
embedding = np.load("results/experiment_001/embeddings/embedding.npy")
print("嵌入形状:", embedding.shape)

# 可视化
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.5)
plt.title('CEBRA Embedding')
plt.show()

# 加载模型（用于新数据）
model = CEBRA.load("results/experiment_001/models/cebra_model.pt")
# new_embedding = model.transform(new_data)
```

### 比较多个实验

创建一个分析脚本来比较不同实验：

```python
import glob
import pandas as pd

# 收集所有实验
experiments = []
for exp_dir in glob.glob("results/*/logs/experiment_results.jl"):
    result = joblib.load(exp_dir)
    experiments.append({
        'name': exp_dir.split('/')[1],
        'dataset': result['config']['dataset'],
        'dim': result['config']['output_dimension'],
        'test_r2': result['metrics']['test_r2']
    })

df = pd.DataFrame(experiments)
print(df.sort_values('test_r2', ascending=False))
```

## 💾 保存到Git

### 添加到版本控制

```bash
# 查看新文件
git status

# 添加实验脚本
git add experiments/

# 提交
git commit -m "Add synthetic data experiment scripts"

# 推送（如果有远程仓库）
git push origin main
```

详见 `GIT_WORKFLOW.md`

## 🔧 故障排除

### 问题1：ImportError: No module named 'cebra'
**解决**：安装CEBRA - `pip install cebra` 或 `pip install -e .`

### 问题2：Python version incompatible
**解决**：确保Python >= 3.9 - `python --version`

### 问题3：CUDA not available
**解决**：使用CPU - `--device cpu`

### 问题4：内存不足
**解决**：
- 减小批次大小：`--batch-size 256`
- 减少迭代：`--max-iterations 1000`
- 使用更小的数据集

### 问题5：训练速度慢
**提示**：
- 确保使用GPU：`--device cuda`
- 检查GPU使用：`nvidia-smi`
- 增加批次大小（如果内存足够）

## 📚 参考资料

- **CEBRA论文**: Schneider, Lee, Mathis (2023) Nature
- **CEBRA文档**: https://cebra.ai/
- **GitHub仓库**: https://github.com/AdaptiveMotorControlLab/CEBRA
- **教程**: 查看 `docs/` 目录

## 🎓 学习路径

1. **开始**: 运行 `quick_test.py` 理解基本流程
2. **探索**: 尝试不同的 `--dataset` 参数
3. **优化**: 调整超参数（维度、迭代次数）
4. **分析**: 比较不同配置的结果
5. **扩展**: 修改脚本适配自己的数据

## ✅ 下一步

- [ ] 完成环境安装（参考 `INSTALL_GUIDE.md`）
- [ ] 运行快速测试验证安装
- [ ] 运行完整实验获取基准结果
- [ ] 尝试不同参数配置
- [ ] 将结果和脚本提交到Git
- [ ] 开始使用自己的数据（需要修改数据加载部分）

## 💡 提示

- 第一次运行会下载数据集（~几MB）
- GPU训练比CPU快10-50倍
- 建议先用小迭代次数测试，确认无误后再长时间训练
- 所有参数都有合理的默认值，可以直接运行
- 结果保存在独立目录中，不会相互覆盖

祝实验顺利！🚀
