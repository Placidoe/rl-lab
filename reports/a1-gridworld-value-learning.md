# A1：GridWorld 的规划与采样学习

## 结果

在自建、确定性的 5×5 GridWorld 中，value iteration 在 9 次 sweep 后收敛；Q-learning 在
3 个独立 seed 的 greedy evaluation 全部到达目标，平均 8 步、平均 return 为 3.0。

$$
3.0 = 7\times(-1)+10
$$

这与从起点到终点的最短 Manhattan 路径长度 8 一致：前 7 步各 -1，进入终点奖励 +10。
该结果只说明最小实现的 Bellman backup、探索与 evaluation 隔离是自洽的；它不是深度 RL、
连续控制或泛化能力的证据。

## 设计

| 项目 | 值 |
| --- | --- |
| 环境 | 自建 deterministic 5×5 GridWorld |
| 折扣 | `\gamma=0.95` |
| 学习率 | `\alpha=0.20` |
| 探索 | `\epsilon` 线性从 1.0 降至 0.02 |
| Q-learning | 6,000 episodes，最多 40 step / episode |
| 评估 | 3 seed，greedy policy，不使用探索噪声 |

## 原理核验

value iteration 是 model-based planning：显式枚举每个动作的后继状态并重复 Bellman backup。
Q-learning 是 model-free control：只利用采到的一条转移 `(s,a,r,s')`。二者在该确定性环境的
最优行为应一致，但得到知识的途径完全不同。

实验中最容易犯的错误是把带 `\epsilon` 的训练轨迹当测试指标。训练故意包含随机探索，因此必须
另用 greedy evaluation 评估当前 policy。

## 产物边界

运行脚本会在被忽略的 `results/a1_gridworld/` 写入临时 seed summaries；本仓库只保存此报告、
代码和下方的脱敏聚合指标。
