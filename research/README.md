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
- 排序指标直接调用固定 `AlphaMaster` 子模块中的 `evaluation_metrics.py`，不另写公式。验证集从 Qlib 获取原始收益标签；训练目标默认使用 CSRankNorm，原始收益及截面超额收益目标是明确记录的实验变量。
- 回测直接调用固定 `FactorVAE` 子模块的 `run_standard_backtest`：Qlib TopkDropoutStrategy、K=30、N=5、risk_degree=0.95、收盘成交、买入 0.0005、卖出 0.0015、min_cost=0；使用各市场原 region、benchmark 和涨跌停设置。
- 扣费只执行一次 `report.return - report.cost`；AR/STD/MDD/Sharpe/Sortino/Calmar 使用 baseline 的 log1p 收益与 252 日年化，ICIR 不年化。每次保存代码哈希和 Qlib 实际版本（当前环境 0.9.7；MASTER 原运行环境记录为 0.9.3.99）。
- 五种子预测必须覆盖相同索引，再以 `avg_none` 平均原始分数，重新计算排序指标并重新回测；不平均种子指标冒充 ensemble。
- 对比全部汇总 baseline，并额外包含 FSDM-results 的结果。记录每项指标最佳值：STD 越低越好，负值 MDD 越接近 0 越好，其余越高越好。

已复算 8 条原 baseline ensemble 净值曲线，六个组合指标与原记录的最大误差为 `1.14e-14`。

另用 FactorVAE 原 seed 42 预测重跑 CSI300 的 727 天及 SP500 的 752 天完整 Qlib 回测，两市场的十项排序/回测指标全部复现，最大差异分别为 `4.0e-15`、`3.9e-15`。可以用 `python -m research.audit_protocol --market csi300 --seed 42 --prediction <原预测CSV>` 重做该检查。

## 搜索与恢复

初始候选包括 Ridge、LightGBM 回归与按交易日分组的排序树、残差网络、市场条件门控、短窗口多尺度时序混合、可学习跨股票因子聚合、BatchEnsemble 和 MASTER 对照。损失比较 MSE、相关性、混合损失、尾部配对排序和 listwise 排序；随后围绕验证集领先方案调整容量、正则、样本时间权重和训练历史范围。

来源：[表格残差网络研究](https://arxiv.org/abs/2106.11959)、[TabM 论文](https://arxiv.org/abs/2410.24210)、[TabM 官方实现](https://github.com/yandex-research/tabm)、[LightGBM 排序目标文档](https://lightgbm.readthedocs.io/en/stable/Parameters.html)、[TimeMixer 官方实现](https://github.com/kwuking/TimeMixer)，以及 `references/papers` 中的 MASTER、FactorVAE、TimeMixer 等论文总结。这里的 BatchEnsemble、时序混合和因子聚合是适配股票任务的实验方案，不宣称完整复现论文模型。

`days_per_update` 比较逐日更新与 4/8 日梯度合并。没有截面归一化或股票间注意力时，多个日期可合并为一次网络前向，但先按日期切开预测，再分别计算排序/分布损失，最后等权平均；不把不同日期的股票作为同一截面。带截面操作的模型仍逐日前向，只合并梯度更新。保存 `batching.json` 及每轮真实更新次数；增大这个参数会减少每轮优化器更新，较长训练、学习率与 patience 的变化是明确记录的实验变量。

数值编码参考 [On Embeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556) 与 [官方实现说明](https://github.com/yandex-research/rtdl-num-embeddings/tree/main/package)。`feature_encoder=ple` 对 158 个股票特征使用训练分位数构建分段线性通道，随后由网络投影学习非线性关系；`periodic` 比较可学习的逐特征 sin/cos 频率。这里保留原始标量通道，编码发生在市场门控之前，76 维上下文继续使用原始标量。这些是本地适配方案，未宣称复现原论文的表格数据结果或证明其股票预测效果。

PLE 默认从当前已剔除边界日、已应用训练历史范围的训练行中，以固定种子抽取最多 65,536 行，完全不读取标签、验证特征或测试特征。同一配置的多个模型种子共用分箱抽样种子。重复/间隔小于 `1e-4` 的分位点合并，常量特征的新增通道置零；外侧通道允许线性外推，再截断到 `[-2,3]`，另保留原始标量通道。边界作为 checkpoint buffer 保存；恢复、验证和测试只加载已有边界，不重新拟合。`feature_encoder.json` 记录抽样行与特征哈希、日期范围、实际每列分箱数和变换参数。

新增风险感知路线：同时预测 5 日收益均值与条件方差，使用 Gaussian NLL 和排序损失联合训练，再比较均值排序、收益/标准差排序及均值减风险惩罚排序。该方案借鉴 [输入相关不确定性研究](https://arxiv.org/abs/1703.04977)，在本任务中的效用需要实验验证。其原始收益训练标签仅取已剔除边界日的训练期，标准差只在这些训练样本上拟合，再截断到 ±8 倍尺度；验证/测试指标仍使用原始收益、原 baseline 公式和策略。预测函数不读取标签，测试阶段不重新拟合尺度。

`factor_gaussian` 借鉴所提供 FactorVAE 论文的动态因子分解，直接学习条件收益的联合高斯分布：协方差为 `diag(exp(diagonal_log_variance)) + B @ B.T`。每只股票预测均值、个股噪声及因子暴露，未来收益只用于训练损失。高斯因子被解析积分，不实现原论文的 VAE posterior/prior 网络；推断只使用已有股票与市场特征。`factor_rank=0` 是独立方差对照，较高维度检验共同冲击是否有助于预测。分布形式见 [PyTorch LowRankMultivariateNormal 文档](https://docs.pytorch.org/docs/2.8/distributions.html#lowrankmultivariatenormal)。

联合似然只在单个交易日内计算，按当天股票数归一化，并与原排序损失加权。用 float64 的小型 Cholesky 分解和后验残差形式计算二次项，避免构建完整股票协方差和大数相减；损失与所有梯度已用完整 MultivariateNormal 对照。评分仍是均值或基于总边际标准差的风险惩罚，原 TopkDropoutStrategy 和费用不变。`covariance_model.json` 保存建模与数值计算约定，验证诊断另记录个股噪声、因子风险及因子方差占比。

`top30_pair` 是单独命名的训练目标：在每个训练日期，将真实收益排名前 30 的股票与其余股票进行确定性的 logistic 配对比较，再以 0.3/0.3/0.4 加权 MSE、相关性损失和配对损失。边界相同收益的股票按比例分配 Top30 归属权重，同收益配对不施加排序偏好；每日单独计算，多个日期合批仍不混合配对。该目标旨在检验持仓区间的排序监督，验证 checkpoint 选择和原 Top30 回测保持原有规则。`ranking_objective.json` 保存其约定；原 `tail_pair` 等目标保持原样。

进一步的对照包括：

- `raw_excess_standardized`：仅对训练日期的真实收益减去当日训练股票截面均值，再拟合训练标准差；预测时不需要当日未来收益或截面真实均值。
- `quantile_aware`：联合预测收益均值和有序的 10%/50%/90% 分位数，用 MSE、pinball loss 和排序损失训练；评分比较均值与下行分位数惩罚。参考 [金融收益条件分位数研究](https://arxiv.org/abs/1308.4276) 和 [神经网络分位数风险学习](https://arxiv.org/abs/2209.06476)，这些来源不构成本股票排序任务有效性的证据。
- `risk_overlay`：收益排序网络按原排序目标训练；验证/测试评分时减去冻结波动模型预测的截面标准化 log variance。冻结来源是同一市场、训练期拟合并按验证集选出的风险模型；其配置和 checkpoint 哈希另存记录。五种子确认改变收益网络的随机种子，冻结风险来源保留原种子，报告必须保留这一差异。

`python -m research.diagnostics --trained research/artifacts/trials/<id>` 只读取剔除边界日的验证集，检查风险预测与绝对误差的日均 RankIC、预测风险分档、80% 区间覆盖率和年份稳定性。新风险/分位数 trial 也会自动保存 `valid/calibration/` 诊断。诊断使用 CPU float32，正式排序与回测仍使用对应 trial 保存的预测；不以诊断分数替换正式预测。超额收益诊断中的真实截面均值只用于定义已实现的验证目标，不参与模型预测。

`scores_blend` 将同市场模型的预测按固定权重组合，比较原始分数、当日截面 z-score 和截面排名归一化。组件配置、权重、归一化方式、平滑系数和模型哈希都写入实验配置/`components.json`。五种子确认会为每个组件设置对应的组合种子，训练缺失组件并复用已完成的组件；每个种子的组合分数仍按 `avg_none` 平均并重新回测。

信号平滑是逐股票的因果 EMA：仅使用当前及更早日期的预测；从当前划分第一天重新开始，超过指定交易日间隔后重置。预测不需要未来真实标签，测试时采用冻结的权重与变换。归一化或平滑会改变模型信号，回测策略及费用不变。自适应搜索约 80% 的提案继续训练模型，约 20% 检验组合，避免廉价的组合实验占满搜索。

checkpoint 仅按验证 RankIC 与年份稳定性选择。整个方法的晋级按验证排序、扣费 AR 和 Sharpe 的百分位排序组合，权重 0.5/0.25/0.25。搜索结束后先写 `selection_lock.json` 冻结方案，再确认 5 个种子并打开测试集。搜索过程不读取新模型的测试表现。

选模锁同时固定 Python 代码包及其 manifest/源码哈希。确认阶段 coordinator 从该包重新启动，所有确认训练、单种子测试和最终集成回测都使用同一代码包；其间的工作区修改不进入最终计算。每次使用前核验包的完整性，测试入口也要求从该包执行。锁中的测试已观测标记按既有新模型测试记录填写，避免把以后可能进行的研究波次误记为首次测试。

测试入口检查配置及种子是否属于冻结方案或其组件。组件的模型哈希发生变化会拒绝组合测试；测试缓存也记录配置、模型哈希与代码哈希。最终比较总是保留十项指标，缺失或非有限值会判定该项未超过 baseline。

`control.json` 可调整研究时长、种子和单任务超时；设 `stop=true` 会结束当前自有训练进程。`extra_candidates.json` 可追加新配置。暂停/终止请求仍需要由控制方明确发出，worker 不会自行根据假设暂停。

运行信息保存在忽略上传的 `research/artifacts/`：

- `STATUS.md`：当前任务和验证结果。
- `study/state.json`、`study/validation_leaderboard.csv`：状态与排行榜。
- `trials/<id>/`：配置、epoch 日志、断点、验证预测、Qlib report、净值和指标。
- `study/selection_lock.json`：打开测试集前的选模记录。
- `holdout/<id>/`、`study/holdout_summary.json`：最终种子/集成结果与 baseline 逐项比较。

worker 使用文件锁，避免重复占用算力。神经网络每轮保存可恢复的模型、优化器和随机状态；完成实验自动跳过。机器或会话停止后，需要在运行环境恢复后重新启动 worker。报告明确区分验证结果、smoke 测试和完整测试结果；失败实验保留日志，不伪造指标。

每次 worker 启动实验时，将当时的 Python 源码保存为带哈希的 `artifacts/code_releases/<hash>/` 包，并从该包运行。`KBS_RESEARCH_WORKSPACE_ROOT` 指定原数据和输出位置；修改工作区源码不会改变已启动进程的代码包。Git 结果快照会连同实际使用的代码包一并导出。

`python -m research.snapshot` 导出小型的 `research/records/20261008/` 验证结果与配置快照，用 Git 保存指标、方法配置、运行环境和代码哈希；大数据、预测和 checkpoint 继续保存在 `artifacts/`。新训练记录 CPU 时间、进程 RSS 峰值和 PyTorch GPU 内存峰值；嵌套组件共享进程，峰值按进程生命周期记录，组合耗时包含缺失组件的训练耗时。

WSL2 主机可另行运行 `python -m research.keep_awake`，通过 Windows 的临时 `ES_SYSTEM_REQUIRED | ES_CONTINUOUS` 请求阻止自动休眠。显示器仍可关闭；研究结束、停止或 heartbeat 失效后释放，不改电源计划。该请求不能阻止手动关机/睡眠，需要电脑持续通电。机制见 [Microsoft 文档](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate)。
