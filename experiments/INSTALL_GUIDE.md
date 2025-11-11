# CEBRA 合成数据实验安装指南

## 问题说明

当前 `conda/cebra_paper.yml` 配置的是 Python 3.8，但最新的 CEBRA 需要 Python 3.9+。

## 解决方案

### 方案1: 使用简化的conda环境（推荐）

创建一个新的环境配置文件：

```bash
cat > conda/cebra_experiment.yml << 'EOF'
name: cebra_experiment
channels:
    - pytorch
    - conda-forge
    - defaults
dependencies:
    - python=3.9
    - pip
    - pytorch::pytorch
    - pytorch::torchvision
    - pytorch::torchaudio
    - pip:
        - cebra
        - joblib
        - scikit-learn
        - scipy
        - matplotlib
        - seaborn
        - numpy
        - pandas
        - tqdm
        - h5py
EOF

# 创建并激活环境
conda env create -f conda/cebra_experiment.yml
conda activate cebra_experiment

# 安装CEBRA (开发模式)
cd /root/cebra_projects/CEBRA-1
pip install -e .
```

### 方案2: 更新现有环境

```bash
conda activate cebra_paper

# 更新Python版本
conda install python=3.9

# 重新安装CEBRA
cd /root/cebra_projects/CEBRA-1
pip install -e .
```

### 方案3: 使用pip安装（最简单）

```bash
# 创建新的虚拟环境
python3.9 -m venv ~/cebra_env
source ~/cebra_env/bin/activate

# 安装依赖
pip install torch torchvision torchaudio
pip install cebra
pip install matplotlib seaborn scikit-learn pandas joblib tqdm h5py

# 或者安装开发版本
cd /root/cebra_projects/CEBRA-1
pip install -e .
```

## 验证安装

```bash
# 测试导入
python -c "import cebra; print(cebra.__version__)"

# 运行快速测试
python experiments/quick_test.py
```

## 运行实验

安装成功后：

```bash
# 快速测试 (1000 iterations, ~5分钟)
python experiments/quick_test.py

# 完整实验 (5000 iterations, ~20分钟)
python experiments/run_synthetic_experiment.py

# 自定义参数
python experiments/run_synthetic_experiment.py \
    --dataset continuous-label-gaussian \
    --max-iterations 3000 \
    --output-dir ./results/my_experiment
```

## 常见问题

### ImportError: No module named 'cebra'
确保已安装CEBRA: `pip install cebra` 或 `pip install -e .`

### CUDA not available
如果没有GPU，使用CPU：`--device cpu`

### AttributeError: module 'numpy' has no attribute 'typeDict'
numpy版本冲突，更新：`pip install numpy==1.24.3`

### Python version not compatible
确保Python >= 3.9: `python --version`
