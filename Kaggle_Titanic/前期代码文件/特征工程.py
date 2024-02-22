import pandas as pd
import numpy as np
import re

#从CSV文件中读取训练和测试数据集
train = pd.read_csv('f:/train.csv')
test = pd.read_csv('f:/test.csv')

#将训练和测试数据集合并到一个列表中，以便于迭代
full_data = [train, test]

freq_port = train.Embarked.dropna().mode()[0]  #在Embarked列中查找最频繁出现的值
print(freq_port)
for dataset in full_data:
    # 用最频繁的值填充Embarked列中缺失的值
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

#将性别映射到数值（0表示男，1表示女）
for dataset in full_data:
    dataset['Sex']=dataset['Sex'].map({'female':1,'male':0}).astype(int)

#从乘客姓名中提取标题
def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)  #使用正则表达式搜索
    #如果搜索到，返回；否则，返回一个空字符串
    if title_search:
        return title_search.group(1)
    return ""

#将get_title函数应用于两个数据集的“Name”列，创建一个新的“title”列
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

#将非常见标题分组为 稀有 类别，并对某些标题进行标准化
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#创建新的“儿童”列并根据年龄设置0/1
for dataset in full_data:
    dataset['Children'] = 1
    for i in range(len(dataset)):
        if dataset['Age'][i] < 18:
            dataset.loc[i, 'Children'] = 0

#创建一个新的“母亲”列，并根据题目年龄、头衔、性别和船上父母/孩子的数量设置0/1
for dataset in full_data:
    dataset['Mother'] = 1
    for i in range(len(dataset)):
        if dataset['Age'][i] >= 18 and dataset['Title'][i] != 'Miss' and dataset['Sex'][i] == 'female' and dataset['Parch'][i] > 0:
            dataset.loc[i, 'Mother'] = 0

#初始化一个2x3矩阵，用于存储基于性别和Pclass的 年龄中值
guess_ages=np.zeros((2,3))

#根据性别和Pclass预测缺失的“年龄”值
for dataset in full_data:
    for i in range(0,2):
        for j in range(0,3):
            guess_df=dataset[(dataset['Sex']==i)&(dataset['Pclass']==j+1)]['Age'].dropna()
            age_guess=guess_df.median()
            guess_ages[i,j]=int(age_guess/0.5+0.5)*0.5
    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[(dataset.Age.isnull())&(dataset.Sex==i)&(dataset.Pclass==j+1),'Age']=guess_ages[i,j]
    dataset['Age']=dataset['Age'].astype(int)

#创建AgeBand年龄阶段这一列，将年龄划分为不同段
train['AgeBand']=pd.cut(train['Age'],5)
#计算AgeBand的平均Survived率
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

#将年龄划分为不同段
for dataset in full_data:
    dataset.loc[dataset['Age']<=16,'Age']=0
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32),'Age']=1
    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age']=2
    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64),'Age']=3
    dataset.loc[dataset['Age']>64,'Age']

for dataset in full_data:
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} #将标记值映射为数值
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0) #用0填充NaN
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int) #将标记值映射为数值

#从训练和测试数据集中删除不必要的列
train=train.drop(['AgeBand'],axis=1)
train_df = train.drop(['Ticket', 'Cabin', 'Parch', 'SibSp', 'Name'], axis=1).copy()
test_df = test.drop(['Ticket', 'Cabin', 'Parch', 'SibSp', 'Name'], axis=1).copy()


train_df.to_csv('f:/processed_train.csv', index=False)
test_df.to_csv('f:/processed_test.csv', index=False)