# CEBRA 合成数据实验脚本

## 快速开始

### 1. 快速测试（推荐先运行）
```bash
cd /root/cebra_projects/CEBRA-1
conda activate cebra_paper
python experiments/quick_test.py
```

这将运行1000次迭代的快速测试，生成 `quick_test_result.png`。

### 2. 完整实验
```bash
python experiments/run_synthetic_experiment.py \
    --dataset continuous-label-poisson \
    --max-iterations 5000 \
    --output-dir ./results/my_experiment
```

## 可用参数

### 数据集选择 (--dataset)
- `continuous-label-poisson` (默认)
- `continuous-label-gaussian`
- `continuous-label-laplace`
- `continuous-label-uniform`
- `continuous-label-t`

### 模型参数
- `--output-dimension 3` - 输出嵌入维度
- `--max-iterations 5000` - 训练迭代次数
- `--batch-size 512` - 批次大小
- `--learning-rate 3e-4` - 学习率
- `--time-offsets 10` - 时间偏移
- `--device cuda` - 使用GPU (或 cpu)

## 输出结果

```
results/
└── synthetic_experiment/
    ├── models/
    │   └── cebra_model.pt
    ├── embeddings/
    │   └── embedding.npy
    ├── figures/
    │   └── embedding_visualization.png
    └── logs/
        ├── experiment_results.jl
        └── summary.txt
```

## 示例实验

### 比较不同噪声类型
```bash
for noise in poisson gaussian laplace; do
    python experiments/run_synthetic_experiment.py \
        --dataset continuous-label-${noise} \
        --max-iterations 3000 \
        --output-dir ./results/noise_${noise}
done
```

### 测试不同嵌入维度
```bash
for dim in 2 3 5 8; do
    python experiments/run_synthetic_experiment.py \
        --output-dimension ${dim} \
        --max-iterations 3000 \
        --output-dir ./results/dim_${dim}
done
```

## 加载结果

```python
import joblib
import numpy as np
from cebra import CEBRA

# 加载模型
model = CEBRA.load("results/synthetic_experiment/models/cebra_model.pt")

# 加载嵌入
embedding = np.load("results/synthetic_experiment/embeddings/embedding.npy")

# 查看指标
results = joblib.load("results/synthetic_experiment/logs/experiment_results.jl")
print(results['metrics'])
```
