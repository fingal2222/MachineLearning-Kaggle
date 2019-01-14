# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 18:18:28 2019

@author: zhaof
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import re

from sklearn import  linear_model
import sklearn.preprocessing as preprocessing

def set_name2(df):
    df["NewName"]=''
    for index,name in enumerate(df.Name.str.findall(pat="(?<=,).*?(?=\.)")):
        df['NewName'][index]=name[0].strip() if len(name)>0 else ''

    return  df


def set_name(df):
    df["Title"]=df.Name.map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
    title_Dict = {}
    title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
    title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
    df["Title"] = df["Title"].map(title_Dict)
#    df.loc[(df.Name.str.contains('Mrs.',regex=False)),'Name'] = 'Mrs'
#    df.loc[(df.Name.str.contains('Mr.',regex=False)),'Name']='Mr'
#    df.loc[(df.Name.str.contains('Miss.',regex=False)), 'Name'] = 'Miss'
#    df.loc[(df.Name.str.contains('Master.',regex=False)), 'Name'] = 'Master'
#    df.loc[~(df.Name.isin(['Mr', 'Miss', 'Mrs','Master'])), 'Name']='other'
    return  df

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin']="Yes"
    df.loc[(df.Cabin.isnull()),'Cabin']='No'
    return df
def set_Age(df):
    age_df=df[['Age','Survived','Fare','Parch','SibSp','Pclass']]
    age_notnull=age_df.loc[df.Age.notnull()]
    age_null=age_df.loc[df.Age.isnull()]
    X=age_notnull.values[:,1:]
    y=age_notnull.values[:,0]
    Rfr=RandomForestRegressor(n_estimators=1000,n_jobs=-1)
    Rfr.fit(X,y)
    age_pred=Rfr.predict(age_null.values[:,1:])
    df.loc[df.Age.isnull(),'Age']=age_pred
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

#数据处理
#测试数据和训练数据统一处理
data_train=pd.read_csv("train.csv")
data_test=pd.read_csv("test.csv")
data_test['Survived']=0
combined_train_test=data_train.append(data_test)
passengerId=data_test['PassengerId']

#处理缺失数据
combined_train_test.Embarked.fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)#众数
combined_train_test=set_name(combined_train_test)
combined_train_test.Fare.fillna(combined_train_test.groupby('Pclass').transform(np.mean).Fare,inplace=True)
combined_train_test["group_ticket"]=combined_train_test.Ticket.groupby(by=combined_train_test.Ticket).transform('count')
combined_train_test['Fare']=combined_train_test['Fare']/combined_train_test["group_ticket"]
combined_train_test.drop(['group_ticket'],axis=1,inplace=True)
combined_train_test['Fare_bin']=pd.qcut(combined_train_test['Fare'],5)
combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]
combined_train_test=set_Age(combined_train_test)
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)
combined_train_test['Ticket_Letter'] = pd.factorize(combined_train_test['Ticket_Letter'])[0]
combined_train_test=set_Cabin_type(combined_train_test)

#one-hot
data_Embarked=pd.get_dummies(combined_train_test.Embarked,prefix='Embarked')
data_Name=pd.get_dummies(combined_train_test.Title,prefix='Name')
data_Cabin=pd.get_dummies(combined_train_test.Cabin,prefix='Cabin')
data_Pclass=pd.get_dummies( combined_train_test.Pclass,prefix='Pclass')
data_Sex=pd.get_dummies( combined_train_test.Sex,prefix='Sex')


df=pd.concat([combined_train_test,data_Pclass,data_Sex,data_Cabin,data_Embarked,data_Name],axis=1)
df.drop(['Pclass','Sex','Cabin','Embarked','Name','Title'],axis=1,inplace=True)

#scaling
scaler=preprocessing.StandardScaler()
age_scale_para=scaler.fit(df.Age.values.reshape(-1,1))
df['Age_scaled']=scaler.fit_transform(df.Age.values.reshape(-1,1),age_scale_para)
df['Fare_scaled']=scaler.fit_transform(df.Fare.values.reshape(-1,1),scaler.fit(df.Fare.values.reshape(-1,1)))

df.to_csv('preprocess.csv',index=False)
#分离测试数据和训练数据
train_data=df[:891]
test_data=df[891:]

#训练模型
X=train_data.drop(['PassengerId','Age','Fare','Survived','Fare_bin','Ticket'],axis=1).values
y=train_data.Survived.values
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# clf=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
clf.fit(X,y)



X_test=test_data.drop(['PassengerId','Age','Fare','Ticket','Fare_bin','Survived'],axis=1).values


predictions=clf.predict(X_test)
result = pd.DataFrame({'PassengerId':passengerId, 'Survived':predictions.astype(np.int32)})
result.to_csv('result.csv',index=False)


train_data=df.drop(['PassengerId','Age','Fare','Survived','Fare_bin','Ticket'],axis=1)
pd.DataFrame({"columns":list(train_data.columns), "coef":list(clf.coef_.T)})








