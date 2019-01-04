# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 18:18:28 2019

@author: zhaof
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
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


##Survived&Fare
#Survived_0 = data_train.Fare[data_train.Survived == 0].value_counts()
#Survived_1 = data_train.Fare[data_train.Survived == 1].value_counts()
#df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
#df.plot(kind='bar', stacked=True)
#plt.title(u"各乘客Fare的获救情况")
#plt.xlabel(u"Fare") 
#plt.ylabel(u"人数") 
#plt.show()

#Survived&Embarked
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客Embarked的获救情况")
plt.xlabel(u"Embarked") 
plt.ylabel(u"人数") 
plt.show()





