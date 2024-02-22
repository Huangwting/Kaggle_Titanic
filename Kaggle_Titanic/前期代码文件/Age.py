import pandas as pd
import numpy as np

#从CSV文件中读取训练和测试数据集
train = pd.read_csv('f:/train.csv')
test = pd.read_csv('f:/test.csv')

#将训练和测试数据集合并到一个列表中，以便于迭代
full_data = [train, test]

#将性别映射到数值（0表示男，1表示女）
for dataset in full_data:
    dataset['Sex']=dataset['Sex'].map({'female':1,'male':0}).astype(int)

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
age_band_survival = train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

#将年龄划分为不同段
for dataset in full_data:
    dataset.loc[dataset['Age']<=16,'Age']=0
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32),'Age']=1
    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age']=2
    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64),'Age']=3
    dataset.loc[dataset['Age']>64,'Age']

#从训练数据集中删除AgeBand
train=train.drop(['AgeBand'],axis=1)

print(train.head(6))