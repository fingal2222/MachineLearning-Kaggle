# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 18:18:28 2019

@author: zhaof
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from sklearn import  linear_model
import sklearn.preprocessing as preprocessing

def set_name2(df):
    df["NewName"]=''
    for index,name in enumerate(df.Name.str.findall(pat="(?<=,).*?(?=\.)")):
        df['NewName'][index]=name[0].strip() if len(name)>0 else ''

    return  df


def set_name(df):
    df.loc[(df.Name.str.contains('Mrs.',regex=False)),'Name'] = 'Mrs'
    df.loc[(df.Name.str.contains('Mr.',regex=False)),'Name']='Mr'
    df.loc[(df.Name.str.contains('Miss.',regex=False)), 'Name'] = 'Miss'
    df.loc[(df.Name.str.contains('Master.',regex=False)), 'Name'] = 'Master'
    df.loc[~(df.Name.isin(['Mr', 'Miss', 'Mrs','Master'])), 'Name']='other'
    return  df

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin']="Yes"
    df.loc[(df.Cabin.isnull()),'Cabin']='No'
    return df

data_train=pd.read_csv("train.csv")
#查看数据情况
data_train.head()
data_train.describe()
data_train.info() #看到有数据缺失
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#survied&Pclass
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级") 
plt.ylabel(u"人数") 
plt.show()

#survived&Sex
Survived_0 = data_train.Sex[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各性别的获救情况")
plt.xlabel(u"乘客性别") 
plt.ylabel(u"人数") 
plt.show()

#Survived&SibSp
Survived_0 = data_train.SibSp[data_train.Survived == 0].value_counts()
Survived_1 = data_train.SibSp[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'Survived_1':Survived_1, u'Survived_0':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客SibSp的获救情况")
plt.xlabel(u"SibSp") 
plt.ylabel(u"人数") 
plt.show()


#Survived&Parch
Survived_0 = data_train.Parch[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Parch[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客Parch的获救情况")
plt.xlabel(u"Parch") 
plt.ylabel(u"人数") 
plt.show()



#Survived&Embarked
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客Embarked的获救情况")
plt.xlabel(u"Embarked") 
plt.ylabel(u"人数") 
plt.show()

#缺失值处理

data_train.Age.fillna(data_train.Age.mean(),inplace=True)
data_train.Cabin=set_Cabin_type(data_train)
data_train=set_name(data_train)
#one-hot
data_Cabin=pd.get_dummies(data_train.Cabin,prefix='Cabin')
data_Pclass=pd.get_dummies( data_train.Pclass,prefix='Pclass')
data_Sex=pd.get_dummies( data_train.Sex,prefix='Sex')

data_Name=pd.get_dummies( data_train.Name,prefix='Name')
# data_SibSp=pd.get_dummies( data_train.SibSp,prefix='SibSp')
# data_Parch=pd.get_dummies( data_train.Parch,prefix='Parch')
data_Embarked=pd.get_dummies(data_train.Embarked,prefix='Embarked')

df=pd.concat([data_train,data_Pclass,data_Sex,data_Embarked,data_Name],axis=1)
df.drop(['Pclass','Sex','Cabin','Embarked','Name'],axis=1,inplace=True)

#scaling
scaler=preprocessing.StandardScaler()
age_scale_para=scaler.fit(df.Age.values.reshape(-1,1))
df['Age_scaled']=scaler.fit_transform(df.Age.values.reshape(-1,1),age_scale_para)
df['Fare_scaled']=scaler.fit_transform(df.Fare.values.reshape(-1,1),scaler.fit(df.Fare.values.reshape(-1,1)))

#训练模型
X=df.drop(['PassengerId','Age','Fare','Survived','Ticket'],axis=1).values
y=df.Survived.values
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# clf=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
clf.fit(X,y)

#处理test数据

data_test=pd.read_csv("test.csv")
data_test=set_name(data_test)
data_test.Cabin=set_Cabin_type(data_test)
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
data_test.Age.fillna(data_test.Age.mean(),inplace=True)

test_Cabin=pd.get_dummies( data_test.Cabin,prefix='Cabin')
test_Pclass=pd.get_dummies( data_test.Pclass,prefix='Pclass')
test_Sex=pd.get_dummies( data_test.Sex,prefix='Sex')
test_Name=pd.get_dummies( data_test.Name,prefix='Name')
# test_SibSp=pd.get_dummies( data_test.SibSp,prefix='SibSp')
# test_Parch=pd.get_dummies( data_test.Parch,prefix='Parch')
test_Embarked=pd.get_dummies(data_test.Embarked,prefix='Embarked')
df1=pd.concat([data_test,test_Pclass,test_Sex,test_Embarked,test_Name],axis=1)
df1.drop(['Pclass','Sex','Cabin','Embarked','Name'],axis=1,inplace=True)

#scaling
scaler=preprocessing.StandardScaler()
age_scale_para1=scaler.fit(df1.Age.values.reshape(-1,1))
df1['Age_scaled']=scaler.fit_transform(df1.Age.values.reshape(-1,1),age_scale_para1)
df1['Fare_scaled']=scaler.fit_transform(df1.Fare.values.reshape(-1,1),scaler.fit(df1.Fare.values.reshape(-1,1)))


X_test=df1.drop(['PassengerId','Age','Fare','Ticket'],axis=1).values


predictions=clf.predict(X_test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv('result.csv',index=False)


train_data=df.drop(['PassengerId','Age','Fare','Survived','Ticket'],axis=1)
pd.DataFrame({"columns":list(train_data.columns), "coef":list(clf.coef_.T)})








