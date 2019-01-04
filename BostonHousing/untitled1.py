# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 19:03:31 2019

@author: zhaof
"""

from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

    
from IPython.display import Image 
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names, 
                         class_names=iris.target_names, 
                         filled=True, rounded=True, 
                         special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data) 
Image(graph.create_png())
    
#os.unlink('iris.dot')