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

#将“Cabin”列值转换为字符串
for dataset in full_data:
    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: str(x))

#将“客舱”列中的“nan”替换为“none”
for dataset in full_data:
    dataset['Cabin'] = dataset['Cabin'].replace('nan', 'none')

#提取“Cabin”值的第一个字符
for dataset in full_data:
    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: x[0] if x != 'none' else 'none')

print(train['Cabin'].head(5))