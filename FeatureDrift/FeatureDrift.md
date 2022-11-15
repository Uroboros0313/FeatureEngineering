# Model Drift
当模型训练成功，被部署在生产环境中后，随着时间的流逝，现实中的条件可能会随着时间而变化，导致测试的数据与训练的数据在性质上出现差异，模型在实际生产环境的性能相比于训练时会降低，这被称为数据漂移。

为了监视数据漂移，有必要从生产环境中收集新数据，并与训练集数据进行比较分析，从而判断是否数据漂移是否在合适的阈值内，否则需要重新收集数据对模型进行重训练，以达到预定的性能。

## 数据/模型漂移的主要来源
- Concept Drift(**概念漂移**): 本质上相同数据的标签随时间的漂移。即上指数据和标签的映射关系发生改变
    $$P_t(y|X) \neq P_{t+1}(y|X)$$
- Data Drift: 数据的分布发生改变
    $$P_t(X) \neq P_{t+1}(X)$$
# Concept Drift检验方法

- DDM
- EDDM
# Data Drift检验方法

1. **Adversarial Validation/Binom Test**

- 步骤
    1. 去掉Label列
    2. 给训练集赋`Drift Label`为0, 测试集赋`Drift Label`为1
    3. 全部放入树模型进行训练, 计算`auc`
    4. `auc`在0.5左右或者对`Drift Label`进行二项检验不显著即不发生数据漂移

2. **Psi**(群体稳定性指标)

    [REF](https://zhuanlan.zhihu.com/p/79682292)

3. **Ks**检验

4. **Chi-Squared**检验(卡方检验)

