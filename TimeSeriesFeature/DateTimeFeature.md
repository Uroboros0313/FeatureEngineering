# Decompose
时间戳数据例子:`2020-08-22-09:00:00`

将时间戳数据分解成:
- 年
- 月
- 日
- 小时
- 早上/下午/晚上/午夜
- 春/夏/秋/冬
- 上旬/中旬/下旬
- 是否周末
- 是否节假日
- 是否月初月末
- 一年的第几个星期
- 一年的第几天
- 一个季度的第几天
- 一周的第几天

# DateWave
一种人工Embedding的方法,起始在科研项目中使用到，后来发现[Ludwig的Date DataType](https://ludwig-ai.github.io/ludwig-docs/0.5/configuration/features/date_features/)同样支持这种形式的Embedding。

## 核心思想

对于日期数据而言，每个Date-Part(月,日,周,天,小时等)除了数值型的含义意外，同样包括一个周期的含义，因此以数字编码和独热编码等方式不合理。
- 数字编码忽略周期性，尤其是在传统机器ML在表格的决策流形下，不适用于拟合这类非线性数据。
- 独热编码带来极大的稀疏性。

DateWave基于每个月的最后一天和第一天实际上距离最接近的假设来理解日期数据。

## 做法

具体如下, 假设每个Date-Part的时间类型是循环的，周期长度为$C_{Day}$，当前相应Part的日期为$d_{Day}$ 。

则编码为:

$$code_{Day} = cos(\frac{d_{Day}}{C_{Day}}\times \pi)$$

在上例中，$C_{Day} = 28/30/31$，当Part是月时，$C_{Month} = 12$，可以根据实际情况调整。

## Code

ref:              
[1] https://blog.csdn.net/deephub/article/details/108415495        
[2] chinese-calendar