# `run_synthetic_experiment.py` 使用手册

本手册介绍 `experiments/run_synthetic_experiment.py` 的整体流程、所依赖的 CEBRA 库组件以及脚本输入输出格式。文末给出使用不同数据集运行实验的要点，帮助快速将脚本迁移到其它数据源上。

## 1. 脚本概览

脚本目标是“端到端”运行一次 CEBRA 合成数据实验，包括：

1. **准备输出目录**：创建 `models/`、`embeddings/`、`figures/`、`logs/` 等子目录。
2. **载入数据集**：通过 `cebra.datasets.init` 获得神经信号与标签。
3. **实例化 CEBRA 模型**：按照命令行参数配置超参数。
4. **训练模型**：调用 `CEBRA.fit` 完成对比学习。
5. **评估与可视化**：将神经数据嵌入到低维空间，并生成图像与线性解码指标。
6. **保存结果**：持久化模型、嵌入、日志和指标。

脚本入口函数 `main()` 负责完成上述流程，并支持 CLI 参数覆盖默认配置。

## 2. 主要依赖与函数解析

### 2.1 标准库与第三方依赖

| 模块 | 作用 |
| --- | --- |
| `argparse` | 解析命令行参数，构建实验配置。 |
| `numpy` | 张量处理、类型兼容 shim（`np.typeDict` 等）确保老版本 SciPy/CEBRA 正常运行。 |
| `matplotlib` | 生成并保存 2D/3D 嵌入及时间序列可视化。 |
| `torch` | CEBRA 底层使用 PyTorch；脚本用它判定 `cuda` 是否可用。 |
| `joblib` | 序列化保存实验配置与指标到 `.jl` 文件。 |
| `pathlib.Path` | 跨平台管理输出目录。 |

### 2.2 CEBRA 库核心组件

| 函数 / 类 | 位置 | 作用与数学含义 |
| --- | --- | --- |
| `cebra.datasets.init(name, download=True)` | `cebra/registry.py` | 通过注册表实例化数据集类。合成数据集如 `continuous-label-poisson` 返回 `SingleSessionDataset`，其中 `dataset.neural` 是形状为 `(T, N)` 的神经活动，`dataset.continuous_index` 是形状为 `(T, L)` 的连续标签。`demo-*` 数据集构造随机张量，因此不需要下载参数。 |
| `cebra.CEBRA(...)` | `cebra/__init__.py` -> `cebra/integrations/sklearn/cebra.py` | 封装 CEBRA 模型。关键超参包括：`model_architecture`（对应 `cebra/models/model.py` 注册的卷积或全连接结构）、`time_offsets`（正样本时间窗口）、`distance`（余弦或欧氏）、`conditional`（`time_delta` / `time`）、`temperature` 等。 |
| `CEBRA.fit(neural, labels)` | 同上 | 训练阶段优化 InfoNCE 对比损失：
  
  `L = - E_{(x_i, x_j^+)} [ log( exp(sim(z_i, z_j^+)/tau) / \sum_k exp(sim(z_i, z_k)/tau) ) ]`
  
  其中 `z_i = f_theta(x_i)` 为嵌入，`sim` 由 `distance` 决定（余弦或负欧氏距离），`tau` 为 `temperature`。正样本对 `(x_i, x_j^+)` 由 `conditional` 和 `time_offsets` 控制。 |
| `CEBRA.transform(neural)` | 同上 | 前向推理，返回形状 `(T, d)` 的嵌入。 |
| `cebra_model.save(path)` | `cebra/integrations/sklearn/cebra.py` | 将模型参数序列化为 `.pt` 文件，可重新加载并继续推理或微调。 |

### 2.3 脚本内部函数

| 函数 | 作用 | 关键实现要点 |
| --- | --- | --- |
| `setup_directories(output_dir)` | 搭建输出目录结构。 | 通过 `Path.mkdir(parents=True, exist_ok=True)` 创建多级目录。 |
| `load_synthetic_data(dataset_name)` | 载入数据并返回 `(neural, labels, dataset)`。 | 对 demo 数据集兼容：若构造器不接受 `download` 参数则再次调用不带该参数的版本。输出的 `neural`、`continuous_labels` 均转换为 NumPy 数组。 |
| `train_cebra_model(neural, labels, config)` | 训练模型。 | 根据 `config` 构造 `CEBRA`，打印配置后调用 `fit`。训练过程中的 tqdm 进度条来自 CEBRA 内部。 |
| `evaluate_embedding(model, neural, labels, output_path)` | 评估嵌入质量并可视化。 | 生成嵌入后调用 `create_visualizations`，再用 `calculate_metrics` 进行线性解码并返回嵌入和指标。 |
| `create_visualizations(embedding, labels, output_path)` | 输出四幅图：2D、3D 嵌入，嵌入随时间变化，标签随时间变化。 | 对标签统一 reshape 为二维数组，避免多维标签导致颜色超范围错误；最多绘制前三个标签维度。图像保存到 `figures/embedding_visualization.png`。 |
| `calculate_metrics(embedding, labels)` | 计算线性解码得分。 | 使用 `sklearn.model_selection.train_test_split` 划分训练/测试集，`Ridge` 回归器拟合标签，返回 train/test R^2 以及 `r2_score`。这里默认标签为连续变量；若数据集只有离散标签，需要改成分类指标。 |
| `save_results(model, embedding, metrics, config, output_path)` | 归档实验结果。 | `model.save` 保存权重，`np.save` 保存嵌入，`joblib.dump` 保存配置与指标，另生成 `summary.txt`。 |

## 3. 输入、输出格式

### 3.1 输入

- **命令行参数**（主要选项）
  - `--dataset`: 数据集名称，脚本当前允许 `continuous-label-*`、`demo-*`。
  - `--model-architecture`: 对应 `cebra/models/model.py` 中注册的架构，如 `offset10-model`、`offset5-model`、`offset1-model` 等。
  - `--output-dimension`: 嵌入维度 `d`。
  - `--max-iterations`: 训练迭代次数。
  - `--batch-size`, `--learning-rate`, `--temperature`, `--time-offsets`, `--distance`, `--conditional`, `--device`：对应 CEBRA 超参。
  - `--output-dir`: 输出目录（默认 `./results/synthetic_experiment`）。

- **数据格式**（由 `cebra.datasets.init` 返回的数据集对象定义）
  - `dataset.neural`: torch.Tensor，形状 `(T, N)`；脚本调用 `.numpy()` 得到浮点数组，`T` 为时间步，`N` 为神经元数。
  - `dataset.continuous_index`: torch.Tensor 或 numpy 数组，形状 `(T,)` 或 `(T, L)`；若数据集仅提供离散标签，可改用 `dataset.discrete_index` 并调整解码器。

### 3.2 输出

成功运行后，`output_dir` 下将产生：

| 路径 | 内容 |
| --- | --- |
| `models/cebra_model.pt` | 训练后的 CEBRA 权重。 |
| `embeddings/embedding.npy` | NumPy 数组，形状 `(T, d)`，即每个时间步的低维嵌入。 |
| `figures/embedding_visualization.png` | 包含 2D/3D 嵌入和时间序列的 PNG 图像。 |
| `logs/experiment_results.jl` | `joblib` 序列化的字典，包含 `config`、`metrics`、`embedding_shape`、`timestamp`。|
| `logs/summary.txt` | 文本摘要，列出配置、指标及嵌入形状。 |

终端输出会显示训练进度、评估指标（train/test/prediction R^2），方便快速查看实验效果。

## 4. 运行方法

### 4.1 默认命令

```bash
python experiments/run_synthetic_experiment.py \
    --dataset demo-continuous \
    --max-iterations 5000 \
    --output-dir ./results/my_experiment
```

脚本会自动检测 GPU；若无 CUDA，请显式指定 `--device cpu`。

### 4.2 切换模型架构

```bash
python experiments/run_synthetic_experiment.py \
    --dataset demo-continuous \
    --model-architecture offset5-model \
    --time-offsets 5 \
    --max-iterations 8000
```

关于各架构：
- `offset10-model`：默认卷积架构，感受野 10。
- `offset5-model`：更浅层，适合短时间依赖或显存受限环境。
- `offset1-model` 系列：纯全连接结构，用于单时刻解码。
- `offset*-mse` / `*-mse-tanh`：输出未归一化或经 `tanh` 压缩的变体，更适合欧氏距离或 MSE 损失。

### 4.3 使用其它数据集

1. **CEBRA 合成数据 (`continuous-label-*`)**
   - 需确保能从 Figshare 下载（默认在 `~/.cebra/synthetic/` 缓存）。若网络限制导致 403，可：
     - 本地运行 `experiments/generate_synthetic_data.py` 生成 `.jl` 文件并放入缓存目录；
     - 改用脚本现已支持的 `demo-*` 或自定义生成的数据。

2. **Demo 随机数据 (`demo-continuous`, `demo-discrete`, `demo-mixed`)**
   - 无需下载，主要用于流程测试。
   - `demo-discrete` / `demo-mixed` 返回离散标签，需要修改 `calculate_metrics` 使用分类模型。

3. **自定义数据**
   - 若已实现新的数据集类并在 `cebra.datasets` 注册（例如 `@register("my-dataset")`），只需在 CLI 中传入对应名称即可。
   - 若数据集提供的标签格式不同（例如多任务、多会话），需在评估和可视化环节对 `continuous_labels` 做相应处理。

4. **真实脑数据（如 Hippocampus、Monkey Reaching）**
   - 这些数据也托管在 Figshare。拿到 `.jl` 或 `.parquet` 后放入 `CEBRA_DATA_PATH` 指定目录即可被 `cebra.datasets.init` 识别。
   - 输出嵌入的解释取决于标签选择：例如行为角度、位置等。线性解码可改为速度、位姿等任务。

## 5. 模型输出解释

- **嵌入 (`embedding.npy`)**：每一行对应输入神经活动窗口的低维表示。可用于可视化、聚类、下游监督解码等。
- **线性解码指标**：脚本默认用 Ridge 回归预测连续标签，输出 `train_r2`、`test_r2`、`prediction_r2`。若标签为周期角度或离散类别，可替换为相应度量（比如 Circular R^2、准确率等）。
- **图像**：帮助诊断嵌入随时间及标签的关系，例如观察轨迹是否平滑、标签是否在嵌入空间中单调变化。

## 6. 扩展与注意事项

- **依赖兼容**：脚本顶部的 NumPy shim (`np.typeDict` 等) 用于在 Python 3.8 + NumPy >= 1.24 环境兼容老版本 SciPy/CEBRA，如升级到 Python 3.9+ 可移除。
- **网络受限**：若无法下载官方数据，请自行生成或放置 `.jl` 文件，同时可修改 `load_synthetic_data` 在失败时回退到本地生成的合成数据（参考 `experiments/quick_test.py`）。
- **多维标签**：`create_visualizations` 只用第一维进行着色，其余维度在时间序列图中显示；若希望更加复杂的可视化，可改用 Plotly 或其它工具。
- **批量实验**：可结合 shell 脚本或 `Makefile` 批量遍历不同超参，并将多个 `results/` 目录汇总分析。

希望本手册能帮助快速理解并扩展 `run_synthetic_experiment.py`。如需进一步自定义（新增数据集、修改损失函数、导出更多指标等），建议阅读 `cebra/models/model.py`、`cebra/datasets/*` 和 `docs/source/usage.rst` 获取更深入的参考。