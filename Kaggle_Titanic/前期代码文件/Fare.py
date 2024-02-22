import pandas as pd

#从CSV文件中读取训练和测试数据集
train = pd.read_csv('f:/train.csv')
test = pd.read_csv('f:/test.csv')

#将训练和测试数据集合并到一个列表中，以便于迭代
full_data = [train, test]

#用中位数进行补全Fare
test['Fare'].fillna(test['Fare'].dropna().median(),inplace=True)

print(train.head(10))