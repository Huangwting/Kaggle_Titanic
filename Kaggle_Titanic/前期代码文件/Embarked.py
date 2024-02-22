import pandas as pd

#从CSV文件中读取训练和测试数据集
train = pd.read_csv('f:/train.csv')
test = pd.read_csv('f:/test.csv')

#将训练和测试数据集合并到一个列表中，以便于迭代
full_data = [train, test]

freq_port=train.Embarked.dropna().mode()[0] #在Embarked列中查找最频繁出现的值
print(freq_port)
for dataset in full_data: #用最频繁的值填充Embarked列中缺失的值
    dataset['Embarked']=dataset['Embarked'].fillna(freq_port)

print(train.head(63))