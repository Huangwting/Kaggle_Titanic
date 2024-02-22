import pandas as pd
import re

#从CSV文件中读取训练和测试数据集
train = pd.read_csv('f:/train.csv')
test = pd.read_csv('f:/test.csv')

#从测试数据集中提取“PassengerId”来使用
PassengerId = test['PassengerId']
print(train.head(3)) #显示训练数据集的前3行

#将训练和测试数据集合并到一个列表中，以便于迭代
full_data = [train, test]

#从乘客姓名中提取标题
def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name) #使用正则表达式搜索
    #如果搜索到，返回；否则，返回一个空字符串
    if title_search:
        return title_search.group(1)
    return ""

#将get_title函数应用于两个数据集的“Name”列，创建一个新的“title”列
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

#计算包括乘客在内的家庭成员总数，并创建“family_num”列
for dataset in full_data:
    dataset['Family_num'] = 1 + dataset['SibSp'] + dataset['Parch']

#创建一个空的“FamilySize”列
for dataset in full_data:
    dataset['FamilySize'] = ''

#根据family_num大小进行分类
for dataset in full_data:
    dataset.loc[dataset['Family_num'] == 1, 'FamilySize'] = '1'
    dataset.loc[(dataset['Family_num'] > 1) & (dataset['Family_num'] <= 4), 'FamilySize'] = '2-4'
    dataset.loc[dataset['Family_num'] > 4, 'FamilySize'] = '>4'

print(train.head(3))