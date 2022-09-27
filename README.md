# FeatureEngineering Operators

## Intro

- Tabular数据集的简单特征工程/预处理算子
   - 时间序列
   - 风控
   - 推荐
   - ... 
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