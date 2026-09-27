# 公开科研数据与环境政策

所有外部数据、环境与 benchmark 必须面向公开科研使用，并在实验开始前记录：

1. 官方来源 URL、版本/commit、许可证或使用条款；
2. 是否允许下载、缓存、再分发与衍生发布；
3. 数据是否含个人信息、受限内容或真实世界可执行动作；
4. train / validation / test 的隔离方式与 hash；
5. 下载时间、内容 hash 和删除策略。

仓库默认只提交下载脚本、来源描述、版本/hash、聚合指标与自建报告；不提交外部原始数据、逐样本
轨迹、模型权重或完整 replay buffer。

## 优先来源

- 在线 RL：Gymnasium、MiniGrid、ALE/Atari 等公开研究环境；先核验各自许可证。
- Offline RL：优先使用 [Minari](https://minari.farama.org/content/basic_usage/) 的版本化公开数据接口。
  D4RL 已说明其环境和 datasets 正迁移到 Gymnasium / Minari，应优先使用新维护路径。
- 更大规模 offline benchmark：可评估 [RL Unplugged](https://deepmind.google/blog/rl-unplugged-benchmarks-for-offline-reinforcement-learning/)；
  它提供 benchmark、数据接口和评估协议，但仍需逐数据集审查条款与硬件需求。

任何来源不清、条款不支持科研使用或不能确认再分发边界的数据，都不进入实验。
