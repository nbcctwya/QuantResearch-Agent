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

`temporal_mixer` 的 `history_steps=8/16/32` 对照扩展历史 Alpha158 窗口，继续使用相同终点日期、股票成员、最新 76 维上下文和原五日收益标签。通过临时加载的原 Qlib sampler 改变 `step_len` 构建独立缓存；随机 512 个窗口与原生 sampler 逐项核对，所有最新 Alpha158 行与原缓存精确核对。原 sampler pickle、8 天缓存和标签缓存保留原样。该路线延续所提供 TimeMixer 总结的多尺度历史建模思路，未实现其完整分解网络；时间混合层的容量随窗口长度增长，作为显式实验变量记录。

`python -m research.data --market csi300 --history-steps 32` 默认只构建训练及验证历史。扩展测试历史要求先存在选模锁，正式测试入口继续核验冻结配置、种子和源码；不在搜索阶段构建长窗口测试缓存。较长缓存不能全部进入显存时，按原日批次读取。`history_model.json` 和缓存 manifest 记录窗口、来源哈希、索引、填充规则及容量约定；原训练/验证末尾五日过滤与已知因果性局限保持原记录。

Qlib 对停牌等缺失行的 `ffill+bfill` 在长短窗口上可能给出不同填充值，因此不能把已填充的 32 天窗口裁成 8 天并声称保留原模型预测。每个时序模型要求自己的原生窗口；冻结收益/风险模型的时序长度不同时拒绝共用窗口。`scores_blend` 仍分别为各组件读取其配置的数据窗口。

数值编码参考 [On Embeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556) 与 [官方实现说明](https://github.com/yandex-research/rtdl-num-embeddings/tree/main/package)。`feature_encoder=ple` 对 158 个股票特征使用训练分位数构建分段线性通道，随后由网络投影学习非线性关系；`periodic` 比较可学习的逐特征 sin/cos 频率。这里保留原始标量通道，编码发生在市场门控之前，76 维上下文继续使用原始标量。这些是本地适配方案，未宣称复现原论文的表格数据结果或证明其股票预测效果。

PLE 默认从当前已剔除边界日、已应用训练历史范围的训练行中，以固定种子抽取最多 65,536 行，完全不读取标签、验证特征或测试特征。同一配置的多个模型种子共用分箱抽样种子。重复/间隔小于 `1e-4` 的分位点合并，常量特征的新增通道置零；外侧通道允许线性外推，再截断到 `[-2,3]`，另保留原始标量通道。边界作为 checkpoint buffer 保存；恢复、验证和测试只加载已有边界，不重新拟合。`feature_encoder.json` 记录抽样行与特征哈希、日期范围、实际每列分箱数和变换参数。

新增风险感知路线：同时预测 5 日收益均值与条件方差，使用 Gaussian NLL 和排序损失联合训练，再比较均值排序、收益/标准差排序及均值减风险惩罚排序。该方案借鉴 [输入相关不确定性研究](https://arxiv.org/abs/1703.04977)，在本任务中的效用需要实验验证。其原始收益训练标签仅取已剔除边界日的训练期，标准差只在这些训练样本上拟合，再截断到 ±8 倍尺度；验证/测试指标仍使用原始收益、原 baseline 公式和策略。预测函数不读取标签，测试阶段不重新拟合尺度。

`factor_gaussian` 借鉴所提供 FactorVAE 论文的动态因子分解，直接学习条件收益的联合高斯分布：协方差为 `diag(exp(diagonal_log_variance)) + B @ B.T`。每只股票预测均值、个股噪声及因子暴露，未来收益只用于训练损失。高斯因子被解析积分，不实现原论文的 VAE posterior/prior 网络；推断只使用已有股票与市场特征。`factor_rank=0` 是独立方差对照，较高维度检验共同冲击是否有助于预测。分布形式见 [PyTorch LowRankMultivariateNormal 文档](https://docs.pytorch.org/docs/2.8/distributions.html#lowrankmultivariatenormal)。

联合似然只在单个交易日内计算，按当天股票数归一化，并与原排序损失加权。用 float64 的小型 Cholesky 分解和后验残差形式计算二次项，避免构建完整股票协方差和大数相减；损失与所有梯度已用完整 MultivariateNormal 对照。评分仍是均值或基于总边际标准差的风险惩罚，原 TopkDropoutStrategy 和费用不变。`covariance_model.json` 保存建模与数值计算约定，验证诊断另记录个股噪声、因子风险及因子方差占比。

`mixture_gaussian` 参考所提供 [MoE 1991 总结](../references/papers/foundations/MoE_1991.md) 中的条件混合似然，以及 [PRISM-VQ 总结](../references/papers/baseline/PRISM_VQ_2026.md) 的条件专家思路。共享股票编码器后，每个分布头预测均值与方差；softmax 门控只使用当前 63 维市场特征，或最新 Alpha158 加市场特征。用混合高斯 NLL 训练各头和门控，随后按混合均值及总方差评分。该本地适配没有实现独立专家网络、VQ 或稀疏路由；权重差异不能作为语义专家分工已成立的证据。

混合似然使用 float64 logsumexp，训练时省去与现有 Gaussian NLL 相同的常量项；默认按 0.7/0.3 加权分布似然与排序损失。总方差包含组件内部方差和组件均值差异。`mixture_components=1` 没有门控，参数及初始化与同配置的 `risk_aware` 相同，构成单高斯对照；float64 损失并不保证优化过程逐位相同。计算形式见 [PyTorch MixtureSameFamily 文档](https://docs.pytorch.org/docs/2.8/distributions.html#mixturesamefamily)，已核对似然、梯度、CDF 和矩。`mixture_model.json` 保存结构、输入与计算约定。

混合分布的 10%/50%/90% 分位数通过其 CDF 二分求逆得到，80% 覆盖率使用真实混合区间。验证诊断记录 PIT 分档、真实收益单位下的负对数密度、门控熵及组件平均权重；标签只在预测生成后用于观测诊断。正式排序、回测和五种子平均仍使用原 baseline 口径。

`student_t` 使用与独立高斯对照相同的两通道网络预测条件均值及完整方差，以固定自由度 `3/5/10` 的 Student-t 似然训练。网络输出的方差为 `exp(log_variance)`；Student-t 分布的尺度平方为该方差乘 `(df-2)/df`，排序风险项继续使用完整标准差。自由度必须大于 2，按配置冻结为 checkpoint buffer。原训练标签标准化及 ±8 截断保持原规则；似然和真实分位数用 float64 计算，验证诊断保留对应精度的真实收益单位均值及标准差。

该假设参考 [Student-t likelihood 研究](https://arxiv.org/abs/2607.25376) 和 [厚尾似然对异常值的研究](https://arxiv.org/abs/2202.03870)，具体计算以 [PyTorch 2.8 StudentT](https://docs.pytorch.org/docs/2.8/distributions.html#studentt) 对照。这里是固定厚尾分布的本地适配，没有引入论文中的贝叶斯权重或 Laplace 分布。12 个新候选交叉比较两个市场、原始/超额收益目标及三个自由度，复用 4 个已完成的单分量高斯对照；两边似然均为 float64，网络初始化、日内排序损失、训练计划及 baseline 回测固定。`return_distribution.json`、真实 Student-t 区间、PIT 与密度诊断保存参数及计算口径。

`top30_pair` 是单独命名的训练目标：在每个训练日期，将真实收益排名前 30 的股票与其余股票进行确定性的 logistic 配对比较，再以 0.3/0.3/0.4 加权 MSE、相关性损失和配对损失。边界相同收益的股票按比例分配 Top30 归属权重，同收益配对不施加排序偏好；每日单独计算，多个日期合批仍不混合配对。该目标旨在检验持仓区间的排序监督，验证 checkpoint 选择和原 Top30 回测保持原有规则。`ranking_objective.json` 保存其约定；原 `tail_pair` 等目标保持原样。

`ema_decay` 比较训练参数的指数移动平均，参考 [PyTorch 2.8 的 AveragedModel 文档](https://docs.pytorch.org/docs/2.8/optim.html#weight-averaging-swa-and-ema) 与 [权重平均研究](https://arxiv.org/abs/1803.05407)。每次训练优化器更新后执行 `EMA = decay * EMA + (1-decay) * raw`，第一次更新直接复制训练权重；原模型继续执行 AdamW 和原 cosine 学习率计划。验证及 best checkpoint 使用 EMA 参数，选模仍只看原验证排序准则；early-stop epoch 的变化属于该方法的实验变量。

训练断点同时保存原模型、EMA 参数/更新次数、优化器、scheduler 和随机状态。已拟合的数值编码与风险状态 buffer 直接复制，不做平均或重新拟合；当前含 BatchNorm 的模型需要另行定义训练统计协议，因此拒绝开启 EMA。`weight_averaging.json` 保存这些约定，最终推断直接加载被选中的普通模型参数。这里没有加入 Mean Teacher 的一致性损失或采用 SWA 的学习率方案；它们的论文结果不构成本任务有效性的证据。

进一步的对照包括：

- `raw_excess_standardized`：仅对训练日期的真实收益减去当日训练股票截面均值，再拟合训练标准差；预测时不需要当日未来收益或截面真实均值。
- `quantile_aware`：联合预测收益均值和有序的 10%/50%/90% 分位数，用 MSE、pinball loss 和排序损失训练；评分比较均值与下行分位数惩罚。参考 [金融收益条件分位数研究](https://arxiv.org/abs/1308.4276) 和 [神经网络分位数风险学习](https://arxiv.org/abs/2209.06476)，这些来源不构成本股票排序任务有效性的证据。
- `risk_overlay`：收益排序网络按原排序目标训练；验证/测试评分时减去冻结波动模型预测的截面标准化 log variance。冻结来源是同一市场、训练期拟合并按验证集选出的风险模型；其配置和 checkpoint 哈希另存记录。五种子确认改变收益网络的随机种子，冻结风险来源保留原种子，报告必须保留这一差异。

`risk_overlay` 的 `risk_regime_strength>0` 对照让风险惩罚随当天预测风险变化。先从所选训练日期的冻结风险预测计算 `s_t=log(mean(exp(log_variance)))`，拟合训练中位数及 `IQR/1.34898` 尺度（下限 0.1），然后使用 `1 + strength * tanh((s_t-center)/(scale*temperature))` 作为惩罚乘数。乘数有明确上下界，统计量作为 checkpoint buffer 保存；恢复、验证和测试不重新拟合。`risk_regime.json` 保存日期范围、统计量哈希、冻结来源和计算约定；`valid/risk_regime/` 记录逐日惩罚及年份均值。

该研究假设参考 [Volatility Managed Portfolios](https://www.nber.org/papers/w22208) 的波动条件风险调整思路。本地实现调整股票评分中的风险惩罚，股票持仓仍由原 TopkDropoutStrategy 计算，risk_degree 和费用保持 baseline 设置；它没有复现该论文的组合收益。统计拟合和预测仅访问已有特征，训练标签、验证标签和测试标签不参与风险状态计算。`risk_regime_strength=0` 保留原静态模型及其 checkpoint 字段。

`risk_overlay` 的 `alpha_source` 对照直接复用指定收益模型的 checkpoint，同时冻结收益与风险网络。只拟合训练特征上的风险状态统计，不再训练收益网络或重新选择收益模型 epoch；这样可在同一收益信号上比较不同惩罚。`fixed_alpha.json` 保存真实来源、epoch 和 checkpoint 哈希，来源被修改时拒绝恢复或测试。五种子确认让收益来源跟随对应种子，缺失来源会按其原配置训练，风险来源继续保留原种子。评分模型的 checkpoint 包含两套网络及已拟合统计，恢复不会重新拟合。零惩罚完整验证复算作为审计单独保存，不参与选模。

`alpha_opportunity_strength>0` 进一步检验收益信号强弱能否调节风险惩罚。只允许冻结收益来源：用所选训练日期的收益预测计算 `a_t=log(max(std(alpha_t, correction=0),1e-6))`，按训练中位数与 IQR 尺度拟合 buffer，再使用 `1 - strength*tanh((a_t-center)/(scale*temperature))`。收益预测更分散时惩罚变轻，更接近时惩罚变重；它可以与原风险状态乘数相乘。乘数有上下界，原静态/风险状态对照及 checkpoint 字段保持兼容。

这里的预测分散度只是待验证的机会代理，不代表已经证明的信号质量。统计拟合只访问冻结模型的训练期特征预测，恢复、验证和测试不重新拟合。`alpha_opportunity.json` 保存真实收益来源种子、checkpoint 哈希、训练日期和统计量哈希；`valid/alpha_opportunity/` 记录每日乘数、总惩罚及年份均值。所有方案仍用原 Top30 回测、持仓比例和费用计算十项指标。

`python -m research.diagnostics --trained research/artifacts/trials/<id>` 只读取剔除边界日的验证集，检查风险预测与绝对误差的日均 RankIC、预测风险分档、80% 区间覆盖率和年份稳定性。新风险/分位数 trial 也会自动保存 `valid/calibration/` 诊断。诊断使用 CPU float32，正式排序与回测仍使用对应 trial 保存的预测；不以诊断分数替换正式预测。超额收益诊断中的真实截面均值只用于定义已实现的验证目标，不参与模型预测。

`scores_blend` 将同市场模型的预测按固定权重组合，比较原始分数、当日截面 z-score 和截面排名归一化。组件配置、权重、归一化方式、平滑系数和模型哈希都写入实验配置/`components.json`。五种子确认会为每个组件设置对应的组合种子，训练缺失组件并复用已完成的组件；每个种子的组合分数仍按 `avg_none` 平均并重新回测。

单来源 `scores_blend` 也可以把按日归一化作为预测方法的一部分：每个种子先生成归一化后的分数，再按 baseline 的 `avg_none` 平均。单种子股票排序保持相同，多个种子的平均排序可能变化。六组真实三种子验证分别做 float64 原样及 `cs_z` 对照，并重新执行 baseline 回测；精度对照的组合指标差异仅为浮点误差，归一化收益则依模型而变化。SP500 的独立风险状态模型改善十项验证指标，CSI300 的无原始通道 PLE 收益模型 AR 下降约 0.0389；两种结果均保存，不把归一化作为所有模型的默认设置。

信号平滑是逐股票的因果 EMA：仅使用当前及更早日期的预测；从当前划分第一天重新开始，超过指定交易日间隔后重置。预测不需要未来真实标签，测试时采用冻结的权重与变换。归一化或平滑会改变模型信号，回测策略及费用不变。自适应搜索约 80% 的提案继续训练模型，约 20% 检验组合或冻结收益信号的评分；冻结评分模型不进入训练候选池。测试缓存要求完整源码哈希匹配，包含数值编码和风险状态变换。

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

18 组混合分布架构/目标对照与 CSI300 的配对三种子复核已完成。两分量模型在每个种子上提高 RankIC 和 AR，但真实 `avg_none` 集成的验证 RankIC 从 `0.07666` 升到 `0.08045`，AR 却从 `0.10765` 降到 `0.04328`。集成预测经过原 Top30 策略后，并不会保留单种子指标的改善，负结果已一并保存。测试阶段仍等待选模锁，超过全部 baseline 的目标尚未得到验证。

长窗口方案通过 66 项 CPU 检查、四个双市场 16/32 天 GPU 案例及精确中断恢复；两个已训练的 8 天模型在各市场五个验证日上给出完全相同的修改前后预测。冻结混合风险模型的双市场零惩罚复算保留完整收益预测，十项 baseline 指标最大差异为 `3.11e-15`。12 组历史长度/数值编码对照和 12 组冻结风险分量/状态评分对照按相同验证规则运行。

EMA 实现通过 72 项 CPU 检查、四类真实 GPU 训练与精确中断恢复，关闭 EMA 后两个市场的训练参数、损失及预测与原源码包完全一致。两个市场的完整关闭 EMA 对照也已完成：全量验证预测、选中参数/epoch 及每轮训练损失均完全一致，十项回测/排序指标最大差异为 `3.56e-15`。八组匹配的完整 EMA 对照继续执行，正确性检查本身不代表模型收益改善。六组三种子评分尺度对照另行记录，不计入完整训练次数或自动晋级。

冻结混合风险评分的八个配对种子补充实验及四组真实 `avg_none` 集成已经完成。SP500 的单分量风险状态评分相对静态评分改善十项验证指标，AR 从 `0.06057` 升到 `0.10806`；CSI300 两分量相对单分量仅小幅改善排序和 STD，AR 从 `0.15414` 降到 `0.14838`，MDD 从 `-0.15127` 变为 `-0.16291`。风险来源保持 seed 0，收益模型按三个种子平均；这项复核没有证明测试集超过 baseline。

加入 Student-t 后的完整回归包含 80 项 CPU 检查，新增两个市场的真实 GPU 训练及精确中断恢复均通过。初次诊断检查的精度差异和修正记录一并保留；通过正确性检查不构成收益改善的证据。

可重做验证汇总：`python -m research.records.20261008.summarize_mixture_validation`、`python -m research.records.20261008.summarize_history_and_risk`、`python -m research.records.20261008.summarize_ema_validation`、`python -m research.records.20261008.summarize_student_validation`。`python -m research.records.20261008.check_mixture_seed_validation` 及 `python -m research.records.20261008.check_mixture_risk_seed_validation` 从三个种子的实际验证预测重新平均并回测，保存来源及 checkpoint 哈希；缺失的种子记录为 pending。冻结风险评分复核中的收益模型跟随种子，风险来源继续共用 seed 0，报告保留这一差异。

结果完整性记录会核对上一版实验和指标、既有种子复核以及每个已保存源码包的哈希。按实际启动包记录复核，早期 56 项实验没有保存不可变启动包的来源，保留当时记录的源码哈希；之前的覆盖计数相差一项，已在完整性记录中纠正。后续实验保留实际启动包，不补造早期运行源码。

导出后、提交前可执行 `python -m research.records.20261008.verify_snapshot_integrity --previous HEAD`；复核已提交快照时用 `--previous <上一快照的提交>` 指定比较起点。该检查要求既有实验、CSV 指标及种子复核记录逐项保留，并核验已记录启动包的全部源码哈希。

WSL2 主机可另行运行 `python -m research.keep_awake`，通过 Windows 的临时 `ES_SYSTEM_REQUIRED | ES_CONTINUOUS` 请求阻止自动休眠。显示器仍可关闭；研究结束、停止或 heartbeat 失效后释放，不改电源计划。该请求不能阻止手动关机/睡眠，需要电脑持续通电。机制见 [Microsoft 文档](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate)。
