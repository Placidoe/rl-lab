# A2：折扣、探索与奖励塑形的受控实验

## 问题

为什么 Q-learning 看似“学不好”？本实验不把所有参数同时改掉，而是分别检验：

1. $gamma$ 改变的是真正的优化目标，还是仅仅改变收敛速度？
2. $epsilon$ 过快衰减如何让行为策略停止发现远期回报？
3. reward shaping 如何帮助学习，同时不偷偷改掉最优策略？

环境有两个终点：远处的 goal 奖励为 $10$，近处的 trap 奖励为 $1$。每个非终止步为 $-1$。

## 数学

Q-learning 仍然拟合同一类 bootstrap target：

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_t+\gamma\max_a Q(s_{t+1},a)-Q(s_t,a_t)\right].
$$

因此，$gamma$ 小并不一定是“训练失败”。它可以让较近的终点拥有更高的折扣回报，因而改变最优行为。

potential-based shaping 使用：

$$
r^{\prime}(s,a,s^{\prime})=r(s,a,s^{\prime})+\gamma\Phi(s^{\prime})-\Phi(s).
$$

本实验令终点的 $\Phi=0$，其余状态为到 goal 的负 Manhattan 距离。整段轨迹的 shaping 项会望远镜求和，
只留下与起点有关的常数；理论上不改变同一 $gamma$ 下的最优策略。

## 运行

```bash
python3 experiments/a2_gridworld_controls/q_learning_control_study.py
```

结果只写入被忽略的 `results/a2_gridworld_controls/`。提交到仓库的是聚合指标、可执行代码、报告和自绘图。
