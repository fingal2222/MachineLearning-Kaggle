# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:55:03 2019

@author: zhaof
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve



#定义衡量标准R2系数 1-sum((y_true-y_pred)**2)/sum((y_true-y_true_mean)**2)0-1之间，数值越大拟合程度越好
def performace_metric(y_true,y_predict):
    from sklearn.metrics import r2_score
    score=r2_score(y_true,y_predict)
    return score

# 根据不同的训练集大小，和最大深度，生成学习曲线
def ModelLearning(X_train,X_test,y_train,y_test,max_depth):    
    m=len(X_train)
    score_train=[]
    score_test=[]
    
    for i in range(1,m,50):
        clf=tree.DecisionTreeRegressor(max_depth=max_depth)
        clf.fit(X_train[0:i],y_train[0:i])
        y_pred=clf.predict(X_train[0:i])
        y_ptest=clf.predict(X_test)
        score_train.append(performace_metric(y_train[0:i],y_pred))
        score_test.append(performace_metric(y_test,y_ptest))
    plt.plot(range(1,m,50),score_train,marker='.',label='training score')
    plt.plot(range(1,m,50),score_test,marker='.',label='test score')
    plt.ylim([0,1.1])
    plt.xlabel("number of training points")
    plt.ylabel("r2_score")
    plt.title("max_depth="+str(max_depth))
    plt.legend()    
    plt.show()
    
def ModelComplexity(X_train,X_test,y_train,y_test,max_depth):
    score_train=[]
    score_test=[]
    for i in max_depth:
        clf=tree.DecisionTreeRegressor(max_depth=i)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_train)
        y_ptest=clf.predict(X_test)
        score_train.append(performace_metric(y_train,y_pred))
        score_test.append(performace_metric(y_test,y_ptest))
    plt.plot(max_depth,score_train,marker='.',label='training score')    
    plt.plot(max_depth,score_test,marker='.',label='test score')
    plt.ylim([0,1.1])
    plt.xlabel("number of training points")
    plt.ylabel("r2_score")
    plt.title("max_depth="+str(max_depth))
    plt.legend()    
    plt.show()       
    
    
    
if __name__=='__main__':
    
    boston=load_boston()
    boston.keys()
    X=pd.DataFrame(boston.data,columns=boston.feature_names)
    y=pd.DataFrame(boston.target,columns=['MEDV'])
    #观察数据情况
    print(X.describe())
    print(X.head())
    
    
    # 目标的最小值，最大值，平均值，中值，标准差
    minimum_price=np.min(y)
    maximum_price=np.max(y)
    meanmum_price=np.mean(y)
    median_price=np.median(y)
    std_price=np.std(y)
    
    print("波士顿房价统计数据：\n")
    
    print ("Minimum price: ",minimum_price)
    print ("Maxmum price: ",maximum_price)
    print ("Meanmum price: ",meanmum_price)
    print ("Median price: ",median_price)
    
    #拆分数据集
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#    ModelLearning(X_train,X_test,y_train,y_test,1)
#    ModelLearning(X_train,X_test,y_train,y_test,3)
#    ModelLearning(X_train,X_test,y_train,y_test,6)
#    ModelLearning(X_train,X_test,y_train,y_test,10)
    
    
    ModelComplexity(X_train,X_test,y_train,y_test,[1,2,3,4,5,6,7,8,9,10])
    
#y_true=[3,-0.5,2,7,4.2]
#y_predict=[2.5,0.0,2.1,7.8,5.3]
#print(performace_metric(y_true,y_predict))

