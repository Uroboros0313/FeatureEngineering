# FeatureEngineering Operators

## Introduction

- Tabular数据集的简单特征工程/预处理算子
   - 时间序列
   - 风控
   - 推荐
   - ... 
 - 本仓库包含的实际上是一系列元算子, 也就意味着复杂的特征工程可以通过组合元算子的方式实现。

## Usage

```python
import pandas as pd

from fe_category import *


# define a op_dict
op_dict = {
    'name': 'LabelEncode',
    'params':{
        'cols':['cat_1']
        'if_prefix': True,
    }
}

df = pd.DataFrame({
    'cat_1': ['c1', 'c2', 'c3', 'c4'],
    'num_1': [1, 2, 1, 3]
})

op_name, op_params = op_dict['name'], op_dict['params'] # get name and params
op_inst = eval(op_name)(op_params)# instantiate an op
op_inst.fit(df)
df = op_inst.transform(df)

```

## TODO

1. 补全特征工程算子
2. 完成元类/父类定义
3. 增加预处理/模型模块, 实现覆盖基本Tabular任务的pipeline

## Rules for collaborators
- Uroboros0313(lisuchi)
  1. 遵守PEP8规范, 尽量编写Type Hints, 我可以不写(**因为我懒**), 但你们要写
  2. 开发在自己的分支进行, 如果不是, 我会回退版本; 本地进行测试后再推入自己的分支或发起merge
  3. 开发版本以dev_作为前缀, merge时去掉前缀
