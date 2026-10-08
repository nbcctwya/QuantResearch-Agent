# QuantResearch-Agent

用于股票收益预测研究的工作区，统一组织实验项目、baseline 代码、实验结果和论文资料。

## 仓库结构

```text
kbs-workspace/
├── kbs-exp/                    # 当前实验项目（子模块）
└── references/
    ├── baseline_code/          # 六个 baseline 仓库（各自为子模块）
    ├── baseline_results/       # baseline 结果仓库（子模块）
    └── papers/                 # 论文 Markdown，由本仓库直接管理
```

| 子模块路径 | GitHub 仓库 | 更新分支 |
|---|---|---|
| `kbs-exp` | [nbcctwya/kbs-exp](https://github.com/nbcctwya/kbs-exp) | `main` |
| `references/baseline_code/AlphaMaster` | [nbcctwya/AlphaMaster](https://github.com/nbcctwya/AlphaMaster) | `main` |
| `references/baseline_code/FactorVAE` | [nbcctwya/FactorVAE](https://github.com/nbcctwya/FactorVAE) | `main` |
| `references/baseline_code/FactorVQVAE` | [nbcctwya/FactorVQVAE](https://github.com/nbcctwya/FactorVQVAE) | `main` |
| `references/baseline_code/MATCC` | [nbcctwya/MATCC](https://github.com/nbcctwya/MATCC) | `master` |
| `references/baseline_code/PRISM-VQ` | [nbcctwya/PRISM-VQ](https://github.com/nbcctwya/PRISM-VQ) | `main` |
| `references/baseline_code/qlib-baselines` | [nbcctwya/qlib-baselines](https://github.com/nbcctwya/qlib-baselines) | `main` |
| `references/baseline_results` | [nbcctwya/baseline_results](https://github.com/nbcctwya/baseline_results) | `main` |

父仓库通过 `.gitmodules` 保存子仓库地址，并记录每个子仓库的固定提交。子模块内的文件与提交历史保存在各自的仓库中，父仓库只管理版本引用。分支配置用于主动更新子模块；普通克隆仍检出父仓库记录的提交。

`PRISM-VQ` 当前引用的是其原远程提交 `d2bb985`。该版本内有一个 `CVQ-VAE` Git 引用，但缺少对应的 `.gitmodules` 地址配置。因此本工作区初始化这八个顶层子模块，不递归初始化 `PRISM-VQ` 内的嵌套引用。

## 克隆与初始化

这些子模块使用 SSH 地址，克隆前需配置 GitHub SSH 访问。

```bash
git clone git@github.com:nbcctwya/QuantResearch-Agent.git kbs-workspace
cd kbs-workspace
git submodule update --init
```

如果已经克隆了父仓库：

```bash
git submodule update --init
```

拉取父仓库更新后，按其记录的版本同步子模块：

```bash
git pull --ff-only
git submodule update --init
```

## 更新一个子模块的引用

以下命令主动获取 `kbs-exp` 更新分支的最新提交，再把新版本记录到父仓库：

```bash
git submodule update --remote -- kbs-exp
git add kbs-exp
git commit -m "Update kbs-exp reference"
git push origin main
```

## 修改子模块代码

普通子模块初始化后通常处于 detached HEAD。修改前先切换到该子仓库的工作分支。以 `kbs-exp` 为例：

```bash
git -C kbs-exp switch main
git -C kbs-exp pull --ff-only
# 修改代码，然后在子仓库提交并推送。
git -C kbs-exp add <修改的文件>
git -C kbs-exp commit -m "Describe the experiment change"
git -C kbs-exp push origin main

# 子仓库推送成功后，在父仓库记录新的提交引用。
git add kbs-exp
git commit -m "Update kbs-exp reference"
git push origin main
```

`MATCC` 的工作分支为 `master`。论文 Markdown 直接在父仓库提交。

## 本地数据

原始 JKP 数据、预生成数据集、checkpoint 和运行缓存由各子仓库的 `.gitignore` 决定；它们不会因为加入父仓库而自动上传。复现实验时，按相应子仓库的说明准备数据与环境。

本仓库忽略本地代理配置、AWS 配置、Python 缓存以及 Windows 下载产生的 `Zone.Identifier` 文件。
