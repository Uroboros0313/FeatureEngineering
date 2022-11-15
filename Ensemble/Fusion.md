# 排序平均法
如果模型评估标准是**与排序或者阈值相关(例如AUC)**, 简单使用平均法并不见得都能取得较好得结果。因为像**波动小的模型**做融合, 对最终结果造成的影响不大。

## 排序法的具体步骤

1. 对预测结果进行排序；
2. 对排序序号进行平均；
3. 对平均排序序号进行归一化。

## 针对新样本排序

针对新样本, 其实有两种处理方式:

1. 重新排序: 将新样本放入原测试集中, 重新排序, 一旦数据量大, 时间复杂度会增加。
2. 参考历史排序: 先将历史测试集的预测结果和排序结果保存, 新样本进来后, 在历史测试集中找到与新样本预测值最近的值, 然后取其排序号赋予新样本。之后平均排序, 使用历史最大最小值进行归一化操作即可。

## 失效场景

1. 预测结果本身方差大
2. 模型间异质性小的情况下
3. 预测AUC较高

满足以上条件, 排序平均法很有可能失效。因为排序平均法会将阈值平均并且**变为均匀分布**。
# Voting
## Hard Voting
多个分类器预测出一个分类, 选择最多的那个分类, 比如预测分类如下:
```py
# 三分类, 每行是一个分类器的预测概率
res = [
    [0.4, 0.1, 0.5],
    [0.35, 0.2, 0.45],
    [0.9, 0.03, 0.07],
]
```
则预测结果为第三类。
## Soft Voting

在Soft Voting中, 将**每个模型的相应的分类的概率作为权值**, 计算就需要将**每个模型对应的类别的概率取平均值**, 然后对比不同的类别的结果。

上述例子中, 软投最后的预测概率是:

```py
res_soft = np.mean(res, axis = 0)
'''
res: [0.55, 0.11, 0.34]
'''
```
即预测结果为第一类

## Sklearn的Voting-Classifier

```py
'''
Example Code
'''
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

'''
Hard Voting
'''
voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC()),
    ('dt_clf', DecisionTreeClassifier(random_state=666),
    ('lgb_clf', LGBMClassifier()))
], voting='hard')

voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)

'''
Soft Voting
'''
voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC(probability=True)),
    ('dt_clf', DecisionTreeClassifier(random_state=666)),
    ('lgb_clf', LGBMClassifier())
], voting='soft')
```
