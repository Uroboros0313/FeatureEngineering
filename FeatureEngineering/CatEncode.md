# CatEncode

- Ref
  
## Woe(风控场景/二分类场景)

- [Ref](https://zhuanlan.zhihu.com/p/146476834)

WOE全称是Weight of Evidence，即证据权重

进行Woe编码前首先要对特征进行分箱(数值型进行分箱)

$$WOE_i = ln(\cfrac{BadCase_i}{BadCase_{total}}) - ln(\cfrac{GoodCase_i}{GoodCase_{total}}) $$

其中$BadCase_i$代表第i个分箱的BadCase数目，total代表所有。

### 为什么数值型变量要做Woe

binning+WOE能解决一个问题，就是可以把非线性的特征转化为线性。

### Woe的好处

- 对波动不敏感。遇到异常数据亦能平稳表现。例如有个人年龄为20，不小心按键盘时按成了200，也不会产生10倍的波动。
- 容易操作。 这是对业务人员说的，用模型计算出评分卡后，给任何不懂技术的人都能算出一个客户的风险值。

## IV(特征筛选)
IV和WOE的差别，就在于IV在WOE基础上乘以一个权重

$$
IV_i = (\cfrac{BadCase_i}{BadCase_{total}} - \cfrac{GoodCase_i}{GoodCase_{total}}) \times WOE_i 
$$

$$
IV = \sum_i IV_i
$$

### WOE的意义
WOE其实描述了变量当前这个分组，对判断个体是否属于坏样本所起到影响方向和大小。WOE值的大小，则是这个影响的大小的体现。
- 当WOE为正时，变量当前取值对判断个体是否会响应起到的正向的影响。
- 当WOE为负时，起到了负向影响。

### IV的意义
IV乘以权重后防止组件样本量不平衡的情况.

### 经验规则
- 若IV信息量取值小于0.02，认为该指标对因变量没有预测能力，应该被剔除；
- 若IV信息量取值在0.02与0.1之间，认为该指标对因变量有较弱的预测能力；
- 若IV信息量取值在0.1与0.3之间，认为该指标对因变量的预测能力一般；
- 若IV信息量取值大于0.3，认为该指标对因变量有较强的预测能力。

## Mean/Target
以类别分组下Label的均值为编码

## Frequency/Count Encoding
以类别出现的频数为编码

## Ordinal Encoding
序列编码，最好是给内部有顺序特征的类别才可以做顺序编码(学历, 年龄等)。