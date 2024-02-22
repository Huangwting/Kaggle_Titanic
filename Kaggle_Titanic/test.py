import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,Perceptron,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

#从CSV文件中读取训练和测试数据集
train = pd.read_csv('f:/processed_train.csv')
test = pd.read_csv('f:/processed_test.csv')

#从训练数据中提取特征 (X_train) 和目标变量 (Y_train)
X_train = train.drop(["Survived"], axis=1)  #去除目标变量列
Y_train = train["Survived"]  #目标变量列

#从测试数据中提取特征 (X_test)
X_test = test

#使用K近邻分类器建立模型
knn = KNeighborsClassifier(n_neighbors=5)  #使用5个近邻
knn.fit(X_train, Y_train)                   #在训练数据上拟合模型
Y_pred = knn.predict(X_test)               #对测试数据进行预测
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)  #计算模型在训练数据上的准确率

#创建包含PassengerId 和预测结果Survived的DataFrame
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_pred})

#将预测结果保存为 CSV 文件
submission.to_csv('f:/knn_predict.csv', index=False)     

#支持向量机分类器（SVC）
svc = SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
acc_svc=round(svc.score(X_train,Y_train)*100,2)
submission=pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":Y_pred})
submission.to_csv('f:/svc_predict.csv',index=False)

#朴素贝叶斯分类器
gaussian = GaussianNB()
gaussian.fit(X_train,Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian=round(gaussian.score(X_train,Y_train)*100,2)
submission=pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":Y_pred})
submission.to_csv('f:/gaussian_predict.csv',index=False)

#决策树分类器
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree=round(decision_tree.score(X_train,Y_train)*100,2)
submission=pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":Y_pred})
submission.to_csv('f:/decision_tree_predict.csv',index=False)

#随机森林分类器
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train,Y_train)
acc_random_forest=round(random_forest.score(X_train,Y_train)*100,2)
submission=pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":Y_pred})
submission.to_csv('f:/random_forest_predict.csv',index=False)

#感知器分类器
perceptron = Perceptron()
perceptron.fit(X_train,Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
submission.to_csv('f:/perceptron_predict.csv',index=False)

#随机梯度下降分类器
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
submission.to_csv('f:/sgd_predict.csv',index=False)

#逻辑回归
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_pred})
submission.to_csv('f:/logreg_predict.csv', index=False)

#线性支持向量分类
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_pred})
submission.to_csv('f:/linear_svc_predict.csv', index=False)

#具有修改损失参数的随机梯度下降（SGD分类器）
sgd_modified = SGDClassifier(loss="modified_huber")
sgd_modified.fit(X_train, Y_train)
Y_pred = sgd_modified.predict(X_test)
acc_sgd_modified = round(sgd_modified.score(X_train, Y_train) * 100, 2)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_pred})
submission.to_csv('f:/sgd_modified_predict.csv', index=False)

#梯度提升机
#可调整参数，例如学习率、最大深度等
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train, Y_train)
Y_pred = gb_classifier.predict(X_test)
acc_gb = round(gb_classifier.score(X_train, Y_train) * 100, 2)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_pred})
submission.to_csv('f:/gradient_boosting_predict.csv', index=False)

#支持向量机的径向基函数核
#可调整参数：C gamma等
svm_rbf = SVC(kernel='rbf', gamma='scale')
svm_rbf.fit(X_train, Y_train)
Y_pred = svm_rbf.predict(X_test)
acc_svm_rbf = round(svm_rbf.score(X_train, Y_train) * 100, 2)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_pred})
submission.to_csv('f:/svm_rbf_predict.csv', index=False)