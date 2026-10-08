# 无人值守研究实验

目标是在 CSI300 和 SP500 的相同测试期、股票池、标签和回测策略下超过现有 baseline。控制文件默认安排 72 小时，其中最后 12 小时开始多种子确认；实际完成时间还取决于最后一项实验和确认任务耗时。只使用本机算力。

## 运行

使用现有 `kbs` 环境。先在工作区根目录生成缓存，再启动唯一的研究 worker：

```bash
/home/nbcctwya/anaconda3/envs/kbs/bin/python -m research.data --market csi300
/home/nbcctwya/anaconda3/envs/kbs/bin/python -m research.data --market sp500
/home/nbcctwya/anaconda3/envs/kbs/bin/python -u -m research.runner
```

worker 需要正常访问 CUDA。当前操作系统的受限沙箱会阻止 GPU，启动训练需使用已授权的沙箱外执行方式。研究代码在父仓库的 `research/` 中；固定的 baseline 子仓库不需要改动。模型使用 `kbs-exp` 的数据接口。

## 评估口径

- 原划分：train 2009–2020、valid 2021–2022、test 2023–2025；标签为 `Ref($close,-5)/Ref($close,-1)-1`。
- 缓存逐窗口对齐现有 Qlib sampler，按日期和股票排序；信息输入与第 234 列未来标签分别存储。新模型直接使用 Alpha158、已有的 13 维 JKP 和 63 维市场信息。
- 训练和验证选模默认剔除各自末尾 5 个交易日，防止目标越过划分边界。测试数据、信号日期和股票覆盖保留原口径。这是公开记录的训练过滤差异；并不表示修复了原 sampler 的全部因果性问题。原来的 `learn` 历史行依赖 DropnaLabel，JKP 的历史修订/发布时间也未独立核验。
- 排序指标直接调用固定 `AlphaMaster` 子模块中的 `evaluation_metrics.py`，不另写公式。验证集从 Qlib 获取原始收益标签；训练目标仍使用 CSRankNorm。
- 回测直接调用固定 `FactorVAE` 子模块的 `run_standard_backtest`：Qlib TopkDropoutStrategy、K=30、N=5、risk_degree=0.95、收盘成交、买入 0.0005、卖出 0.0015、min_cost=0；使用各市场原 region、benchmark 和涨跌停设置。
- 扣费只执行一次 `report.return - report.cost`；AR/STD/MDD/Sharpe/Sortino/Calmar 使用 baseline 的 log1p 收益与 252 日年化，ICIR 不年化。每次保存代码哈希和 Qlib 实际版本（当前环境 0.9.7；MASTER 原运行环境记录为 0.9.3.99）。
- 五种子预测必须覆盖相同索引，再以 `avg_none` 平均原始分数，重新计算排序指标并重新回测；不平均种子指标冒充 ensemble。
- 对比全部汇总 baseline，并额外包含 FSDM-results 的结果。记录每项指标最佳值：STD 越低越好，负值 MDD 越接近 0 越好，其余越高越好。

已复算 8 条原 baseline ensemble 净值曲线，六个组合指标与原记录的最大误差为 `1.14e-14`。

另用 FactorVAE 原 seed 42 预测重跑 CSI300 的 727 天及 SP500 的 752 天完整 Qlib 回测，两市场的十项排序/回测指标全部复现，最大差异分别为 `4.0e-15`、`3.9e-15`。可以用 `python -m research.audit_protocol --market csi300 --seed 42 --prediction <原预测CSV>` 重做该检查。

## 搜索与恢复

初始候选包括 Ridge、LightGBM 回归与按交易日分组的排序树、残差网络、市场条件门控、短窗口多尺度时序混合、可学习跨股票因子聚合、BatchEnsemble 和 MASTER 对照。损失比较 MSE、相关性、混合损失、尾部配对排序和 listwise 排序；随后围绕验证集领先方案调整容量、正则、样本时间权重和训练历史范围。

来源：[表格残差网络研究](https://arxiv.org/abs/2106.11959)、[TabM 论文](https://arxiv.org/abs/2410.24210)、[TabM 官方实现](https://github.com/yandex-research/tabm)、[LightGBM 排序目标文档](https://lightgbm.readthedocs.io/en/stable/Parameters.html)、[TimeMixer 官方实现](https://github.com/kwuking/TimeMixer)，以及 `references/papers` 中的 MASTER、FactorVAE、TimeMixer 等论文总结。这里的 BatchEnsemble、时序混合和因子聚合是适配股票任务的实验方案，不宣称完整复现论文模型。

checkpoint 仅按验证 RankIC 与年份稳定性选择。整个方法的晋级按验证排序、扣费 AR 和 Sharpe 的百分位排序组合，权重 0.5/0.25/0.25。搜索结束后先写 `selection_lock.json` 冻结方案，再确认 5 个种子并打开测试集。搜索过程不读取新模型的测试表现。

`control.json` 可调整研究时长、种子和单任务超时；设 `stop=true` 会结束当前自有训练进程。`extra_candidates.json` 可追加新配置。暂停/终止请求仍需要由控制方明确发出，worker 不会自行根据假设暂停。

运行信息保存在忽略上传的 `research/artifacts/`：

- `STATUS.md`：当前任务和验证结果。
- `study/state.json`、`study/validation_leaderboard.csv`：状态与排行榜。
- `trials/<id>/`：配置、epoch 日志、断点、验证预测、Qlib report、净值和指标。
- `study/selection_lock.json`：打开测试集前的选模记录。
- `holdout/<id>/`、`study/holdout_summary.json`：最终种子/集成结果与 baseline 逐项比较。

worker 使用文件锁，避免重复占用算力。神经网络每轮保存可恢复的模型、优化器和随机状态；完成实验自动跳过。机器或会话停止后，需要在运行环境恢复后重新启动 worker。报告明确区分验证结果、smoke 测试和完整测试结果；失败实验保留日志，不伪造指标。

WSL2 主机可另行运行 `python -m research.keep_awake`，通过 Windows 的临时 `ES_SYSTEM_REQUIRED | ES_CONTINUOUS` 请求阻止自动休眠。显示器仍可关闭；研究结束、停止或 heartbeat 失效后释放，不改电源计划。该请求不能阻止手动关机/睡眠，需要电脑持续通电。机制见 [Microsoft 文档](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate)。
