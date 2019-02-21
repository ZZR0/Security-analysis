# Author:ZZR

import numpy as np
from sklearn.model_selection import train_test_split as ts
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.model_selection import  validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# X = list()
# y = list()
#
# with open('breast_cancer_train.txt','r',encoding='utf-8') as file:
# 	for line in file.readlines():
# 		line = line.strip().split('\t')
# 		X.append(list(map(float,line[:-1])))
# 		y.append(line[-1])
#
# X = np.array(X)
# y = np.array(y)
# # # print(X)
# # # print(y)
# #
# # X = preprocessing.normalize(X)
# # # print(X)
# # X = preprocessing.scale(X)
# #
# X_train,X_test,y_train,y_test = ts(X,y,test_size=0.2)
# #
# #
# # # kernel = 'rbf'
# # clf_rbf = SVC(C= 1, kernel='rbf', tol = 0.001)
# # clf_rbf.fit(X_train,y_train)
# # score_rbf = clf_rbf.score(X_test,y_test)
# # print("The score of rbf is : %f"%score_rbf)
# # joblib.dump(clf_rbf, 'save/clf_rbf.pkl')
# # clf3 = joblib.load('save/clf_rbf.pkl')
# # print(clf3.score(X,y))
#
# param_range = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
# train_accuracy, test_accuracy = validation_curve(
#         SVC(kernel='rbf'), X, y, param_name='C', param_range=param_range, cv=10,
#         scoring='accuracy')
# train_accuracy_mean = np.mean(train_accuracy, axis=1)
# test_accuracy_mean = np.mean(test_accuracy, axis=1)
#
# plt.plot(param_range, train_accuracy_mean, 'o-', color="r",
#              label="Training")
# plt.plot(param_range, test_accuracy_mean, 'o-', color="g",
#              label="Cross-validation")
#
# plt.xlabel("C")
# plt.ylabel("Accuracy")
# plt.legend(loc="best")
# plt.show()
# # kernel = 'linear'
# clf_linear = svm.SVC(C= 1, kernel='linear')
# clf_linear.fit(X_train,y_train)
# score_linear = clf_linear.score(X_test,y_test)
# print("The score of linear is : %f"%score_linear)
#
# # kernel = 'poly'
# clf_poly = svm.SVC(C= 1, kernel='poly')
# clf_poly.fit(X_train,y_train)
# score_poly = clf_poly.score(X_test,y_test)
# print("The score of poly is : %f"%score_poly)

# for i in range(5,101,5):
# 	print(i)

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:37:21 2015

@author: Eddy_zheng
"""
from mpl_toolkits.mplot3d import Axes3D

data = np.random.randint(0, 255, size=[40, 40, 40])

x, y, z = data[0], data[1], data[2]
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(x[:10], y[:10], z[:10], c='y')  # 绘制数据点
ax.scatter(x[10:20], y[10:20], z[10:20], c='r')
ax.scatter(x[30:40], y[30:40], z[30:40], c='g')

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()
