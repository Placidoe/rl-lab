# RL 实验 Todo

每一个 checkbox 只有在代码、配置、聚合指标、Markdown 报告和原创原理图齐全后才可以勾选。

## A. Foundations

- [ ] A1 GridWorld：实现 policy evaluation、value iteration、Q-learning。
- [ ] A2 比较 `\gamma`、`\epsilon`、奖励塑形；解释收敛失败。
- [ ] A3 图解 Bellman backup、探索与 state visitation。

## B. Deep RL

- [ ] B1 CartPole DQN：replay、target network、定期 evaluation。
- [ ] B2 Double / Dueling / PER 消融；只改一个变量。
- [ ] B3 LunarLander PPO：GAE、clip、entropy、KL 仪表盘。
- [ ] B4 多 seed 汇总：均值、标准差、置信区间与 sample efficiency。

## C. 环境与动作空间

- [ ] C1 MiniGrid：部分可观测、记忆与探索。
- [ ] C2 连续控制：SAC 与 TD3 的稳定性对照。
- [ ] C3 视觉输入：frame stack、CNN policy、环境吞吐 profiling。

## D. 数据驱动与偏好

- [ ] D1 Behavior Cloning baseline。
- [ ] D2 Offline RL：IQL 或 CQL，单独检测 OOD action 与 conservative bias。
- [ ] D3 偏好 reward model：合成偏好、held-out ranking、reward hacking 反例。

## E. LLM Agent RL

- [ ] E1 可验证奖励 toy environment；不以自评做真值。
- [ ] E2 rollout / verifier / advantage / update 的最小闭环。
- [ ] E3 对照 SFT、拒绝采样与 group-based optimization。
- [ ] E4 只在通过多 seed、独立验证器与 held-out task 后扩展规模。

## F. 系统实验

- [ ] F1 同一算法下测单环境与 vectorized rollout 吞吐。
- [ ] F2 测 batch size、sequence length、GPU utilization、CPU 饱和度。
- [ ] F3 T4×2：单卡、DDP、模型/采样分工的公平对照。

## 统一发布门槛

- [ ] 源码可从干净环境运行。
- [ ] 原始数据、逐样本轨迹、权重和临时结果不提交。
- [ ] 报告写清因果变量、失败案例和不可推广的范围。
