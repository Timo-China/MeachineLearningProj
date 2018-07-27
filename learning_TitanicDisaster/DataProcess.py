'''
主要更加图表分析后对数据进行适当处理，比如缺失数据等

1.如果缺值的样本占总数比例极高，我们可能就直接舍弃了，作为特征加入的话，可能反倒带入noise，影响最后的结果了
2.如果缺值的样本适中，而该属性非连续值特征属性(比如说类目属性)，那就把NaN作为一个新类别，加到类别特征中
3.如果缺值的样本适中，而该属性为连续值特征属性，有时候我们会考虑给定一个step(比如这里的age，我们可以考虑每隔2/3岁为一个步长)，然后把它离散化，之后把NaN作为一个type加到属性类目中。
有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上。

'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
import re
import time

'''
从名字中获取头衔，比如miss、 Mr、Master有时候也能象征不同人的身份地位
'''
def Get_Title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def SetMissingAge(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    print(predictedAges.size)


    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr

def SetCabinType(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

def PreprocessingData(data_train):

    # 从姓名中提取人的头衔
    data_train['Title'] = data_train['Name'].apply(Get_Title)

    data_train['Title'] = data_train['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data_train['Title'] = data_train['Title'].replace('Mlle', 'Miss')
    data_train['Title'] = data_train['Title'].replace('Ms', 'Miss')
    data_train['Title'] = data_train['Title'].replace('Mme', 'Mrs')

    # 添加FamilySize
    data_train['FamilySize'] = data_train['SibSp'] + data_train['Parch'] + 1

    data_train, rfr = SetMissingAge(data_train)
    # Cabin缺失项太多，直接除去Cabin项后，精度进一步提升,得分为0.78468
    # data_train = SetCabinType(data_train)

    data_train['Age_bin'] = pd.cut(data_train['Age'], bins=[0, 12, 20, 40, 120], labels=['Children', 'Teenage', 'Adult', 'Elder'])
    data_train['Fare_bin'] = pd.cut(data_train['Fare'], bins=[0, 7.91, 14.45, 31, 120], labels=['Low_fare', 'median_fare', 'Average_fare', 'high_fare'])

    # 转为onehot
    # dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    dummies_age = pd.get_dummies(data_train['Title'], prefix='Age_bin')
    dummies_fare = pd.get_dummies(data_train['Title'], prefix='Fare_bin')
    dummies_title = pd.get_dummies(data_train['Title'], prefix='Title')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
    df = pd.concat([data_train, dummies_title, dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_age, dummies_fare], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age_bin', 'Fare_bin'], axis=1, inplace=True)

    # 将Age与Fare划分等级后，精度进一步提升
    # scaler = preprocessing.StandardScaler()
    # age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
    # df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
    # fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
    # df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

    # 用正则取出我们要的属性值
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values
    return train_np

def savePredictResult(test_data, predict, csv_path):
    df_svm_predict = pd.DataFrame(predict, columns=['Survived'])
    result = pd.concat([test_data, df_svm_predict], axis=1)
    result = result[['PassengerId', 'Survived']]
    result.to_csv(csv_path, index=False)

def main():

    # read train data
    data_train = pd.read_csv('Data/train.csv')

    # process training data
    train_np = PreprocessingData(data_train)

    # get train data survived class
    y = train_np[:, 0]

    # get all features
    X = train_np[:, 1:]

    # use logistic Regression
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    # predict test data
    test_data = pd.read_csv('Data/test.csv')
    # test_data.loc[(test_data.Fare.isnull()), 'Fare'] = 0
    test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

    test_np = PreprocessingData(test_data)
    test_x = test_np
    predict_y = clf.predict(test_x)

    print('predict value:', predict_y)

    # 5 cv
    print(cross_val_score(clf, X=X, y=y, cv=5))

    split_train_x, split_test_x, split_train_y, split_test_y = train_test_split(X, y, test_size=0.3, random_state=6)

    clf2 = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf2.fit(split_train_x, split_train_y)

    predict_y = clf2.predict(split_test_x)
    acc = accuracy_score(y_true=split_test_y, y_pred=predict_y)
    print('Acc:', acc)

    # use SVM, 结果上传kaggle得了0分(┬＿┬)
    from sklearn.svm import SVC
    svc_clf = SVC()
    svc_clf.fit(X=split_train_x, y=split_train_y)
    print('svm score:', svc_clf.score(split_test_x, split_test_y))
    svm_predict_y = svc_clf.predict(split_test_x)
    print('predict value:', svm_predict_y)
    print('svm accuracy', accuracy_score(y_true=split_test_y, y_pred=svm_predict_y))
    # svm_predict_test_y = svc_clf.predict(test_x)
    # savePredictResult(test_data, svm_predict_test_y, 'Data/result.csv')

    #votingclassifier 结果上传kaggle 得0分
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.ensemble import RandomForestClassifier

    vot_clf = VotingClassifier(
        estimators=[
            ('logistic_clf', linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)),
            ('svm_clf', SVC()),
            ('decisiontree_clf', DecisionTreeClassifier()),
            ('radom_forest', RandomForestClassifier(random_state=666))
        ], voting='hard'
    )
    # vot_clf.fit(X=X, y=y)
    # print('voting score:', vot_clf.score(split_test_x, split_test_y))
    # vot_predict = vot_clf.predict(test_x)
    # print('predict value:', vot_predict)
    # savePredictResult(test_data, vot_predict, 'Data/result.csv')

    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
    #                                bootstrap_features=False, n_jobs=-1)

    # 使用此参数分数可以提高到0.77990，使用svm算法做评估器比线性回归更加准确一点，下面可以进行调节SVM的参数
    bagging_clf = BaggingClassifier(svc_clf, n_estimators=1000, max_samples=0.5, max_features=1.0, bootstrap=True,oob_score=True,
                                   bootstrap_features=False, n_jobs=-1)

    bagging_clf.fit(X, y)
    print(bagging_clf.oob_score_)
    bagging_predict = bagging_clf.predict(test_x)
    bagging_predict = bagging_predict.astype(np.int)
    file_name = time.strftime('Data/Result_%Y_%m_%d_%H_%M_%S.csv', time.localtime())
    savePredictResult(test_data, bagging_predict, file_name)
    print('finish save')


if __name__ == '__main__':
    main()


