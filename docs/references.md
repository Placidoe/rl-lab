# RL 阅读与论文路线

按实验顺序阅读；每篇论文都要回答“它修复了哪种失败模式、代价是什么、怎样在实验中观测到”。

## 基础

1. Sutton & Barto，《[Reinforcement Learning: An Introduction](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)》：MDP、动态规划、MC、TD、eligibility traces。
2. Mnih et al.，[DQN](https://www.nature.com/articles/nature14236)：高维观测上的 replay buffer 与 target network。
3. Schulman et al.，[GAE](https://arxiv.org/abs/1506.02438)：advantage estimator 的 bias--variance tradeoff。
4. Schulman et al.，[PPO](https://arxiv.org/abs/1707.06347)：clipped surrogate 与 on-policy 多 epoch update。

## 连续控制与 Offline RL

1. Haarnoja et al.，[SAC](https://arxiv.org/abs/1801.01290)：maximum-entropy actor-critic。
2. Kumar et al.，[CQL](https://arxiv.org/abs/2006.04779)：对 OOD action 的 conservative Q regularization。
3. Kostrikov et al.，[IQL](https://arxiv.org/abs/2110.06169)：隐式 value learning 与 advantage-weighted BC。
4. Fu et al.，[D4RL](https://arxiv.org/abs/2004.07219)：固定 logged dataset 的 benchmark 设计。

## 偏好与 LLM Agent RL

1. Ouyang et al.，[InstructGPT / RLHF](https://arxiv.org/abs/2203.02155)：demonstration、preference、reward model、PPO 的链路。
2. 先做可验证奖励环境，再研究人类偏好：verifier 是评估真值；reward model 是会有偏差的近似器。

## 阅读产出模板

每篇论文写一页 Markdown：

- 要解决的 failure mode；
- 目标函数逐项含义；
- assumptions 与数据/环境边界；
- 最小可复现实验；
- 至少一个预期失败反例；
- 与前一算法的公平对照。
