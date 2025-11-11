# Git工作流程指南

## 保存新增的实验脚本

您已经创建了新的实验脚本，以下是如何保存和管理它们的步骤。

### 1. 查看当前改动

```bash
cd /root/cebra_projects/CEBRA-1
git status
```

### 2. 添加新文件到Git

```bash
# 添加experiments目录下的所有文件
git add experiments/

# 或者单独添加每个文件
git add experiments/run_synthetic_experiment.py
git add experiments/quick_test.py
git add experiments/README.md
git add experiments/INSTALL_GUIDE.md
git add experiments/GIT_WORKFLOW.md
```

### 3. 提交改动

```bash
git commit -m "Add synthetic data experiment scripts

- run_synthetic_experiment.py: Complete experiment pipeline
- quick_test.py: Quick test script for validation
- README.md: Usage instructions
- INSTALL_GUIDE.md: Installation guide
- GIT_WORKFLOW.md: Git workflow instructions"
```

### 4. 推送到远程仓库

#### 如果有写权限（自己的fork或私有仓库）：

```bash
# 推送到主分支
git push origin main

# 或推送到特定分支
git checkout -b experiments/synthetic-data
git push origin experiments/synthetic-data
```

#### 如果没有写权限（原始仓库）：

1. **Fork原仓库**到您自己的GitHub账号

2. **添加您的fork为远程仓库**：
```bash
git remote add myfork https://github.com/YOUR_USERNAME/CEBRA.git
```

3. **推送到您的fork**：
```bash
git push myfork main
# 或
git push myfork experiments/synthetic-data
```

4. **在GitHub上创建Pull Request**（可选）

### 5. 下次继续工作

**在同一台机器上**：

```bash
cd /root/cebra_projects/CEBRA-1
git pull origin main  # 或 git pull myfork main
# 继续修改和工作
```

**在新机器上**：

```bash
# Clone您的远程仓库
git clone https://github.com/YOUR_USERNAME/CEBRA.git
cd CEBRA

# 设置环境并运行实验
conda env create -f conda/cebra_experiment.yml
conda activate cebra_experiment
python experiments/quick_test.py
```

## 常用Git命令

```bash
# 查看状态
git status

# 查看改动
git diff

# 查看历史
git log --oneline

# 撤销未提交的改动
git checkout -- <file>

# 暂存改动（不提交）
git stash
git stash pop  # 恢复暂存

# 创建新分支
git checkout -b new-branch-name

# 切换分支
git checkout main

# 合并分支
git merge other-branch
```

## 最佳实践

1. **定期提交**：完成一个功能后立即提交
2. **有意义的提交信息**：清楚描述做了什么改动
3. **使用分支**：不同实验或功能使用不同分支
4. **推送到远程**：避免只在本地保存代码
5. **保持同步**：开始工作前先 `git pull`

## 检查远程仓库配置

```bash
# 查看远程仓库
git remote -v

# 查看当前分支
git branch

# 查看所有分支（包括远程）
git branch -a
```

## 示例工作流

```bash
# 1. 确保在最新版本
git pull origin main

# 2. 创建实验分支
git checkout -b experiments/new-feature

# 3. 进行修改和实验
python experiments/run_synthetic_experiment.py --dataset continuous-label-gaussian

# 4. 添加和提交
git add experiments/
git commit -m "Add gaussian noise experiment results"

# 5. 推送到远程
git push origin experiments/new-feature

# 6. 继续工作或创建PR
```
