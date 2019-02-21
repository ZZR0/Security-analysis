# Author:ZZR

import xlrd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as ts
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadData(path):
	name, X, y, mlp_y, change = [], [], [], [], []
	
	
	for p in path:
		# pX, py, ch, pmlpy = [], [], [], [[],[],[],[],[],[],[],[]]
		pname, pX, py, ch, pmlpy = [], [], [], [], []
		
		book = xlrd.open_workbook(p)
		data =book.sheet_by_index(0)

		row = data.nrows
	
		for i in range(1,row):
			# mlplabel = []
			row_values = data.row_values(i)
			stockid = row_values[1]
			stocklabel = row_values[17]
			stockchange = row_values[19]
			mlplabel = row_values[25]
			# for j in range(8):
			# 	# mlplabel.append()
			# 	pmlpy[j].append(row_values[25+j])
			pname.append(stockid)
			pX.append(row_values[2:17])
			py.append(stocklabel)
			ch.append(stockchange)
			pmlpy.append(mlplabel)
			
			# print(stockid,':',stocklabel)
	
		pX = np.array(pX)
		py = np.array(py)
		ch = np.array(ch)
		pmlpy = np.array(pmlpy)
		
		pX = preprocessing.scale(pX)
		# pX = preprocessing.normalize(pX)
		
		name.append(pname)
		X.append(pX)
		y.append(py)
		change.append(ch)
		mlp_y.append(pmlpy)
	
	return name, X, y, change, mlp_y


def validation_curve(X, y):
	param_range = [1,2,3,4,5,6,7,8]
	
	r = [(10),(20,10),(30,20,10),]
	
	train_accuracy_mean = []
	test_accuracy_mean = []
	
	first = []
	second = []
	score = []
	
	for i in range(5,51,5):
	
		score_rbf_train = []
		score_rbf_test = []
		print(i)
		for j in range(5,51,5):
			X_train, X_test, y_train, y_test = ts(X, y, test_size=0.2)
			print(j)
			# clf2016_rbf = SVC(C=0.9, kernel='poly', degree=i)
			# clf2016_rbf.fit(X_train, y_train)
			#
			# score_rbf_train.append(clf2016_rbf.score(X, y))
			# score_rbf_test.append(clf2016_rbf.score(X_test, y_test))
		
			mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(i, j))
			mlp.fit(X_train, y_train)
		
			# score_rbf_train.append(mlp.score(X, y[i-1]))
			# score_rbf_test.append(mlp.score(X_test, y_test))
			
			first.append(i)
			second.append(j)
			score.append(mlp.score(X_test, y_test))
			
		# train_accuracy_mean.append(np.mean(score_rbf_train))
		# test_accuracy_mean.append(np.mean(score_rbf_test))
	
	# plt.plot(param_range, train_accuracy_mean, 'o-', color="r",
	#              label="Training")
	# plt.plot(param_range, test_accuracy_mean, 'o-', color="g",
	#              label="Cross-validation")
	#
	# plt.xlabel("degree")
	# plt.ylabel("Accuracy")
	# plt.legend(loc="best")
	
	ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
	#  将数据点分成三部分画，在颜色上有区分度
	ax.scatter(first, second, score, c='r')  # 绘制数据点
	ax.set_zlabel('Accuracy')  # 坐标轴
	ax.set_ylabel('second')
	ax.set_xlabel('first')
	
	plt.show()

def grading(i, forecast, change):
	buy = []
	overweight = []
	middle = []
	underweight = []
	sell = []
	# if i == 1:
	# 	print(forecast)
	for j in range(len(forecast)):
		if forecast[j][0] > 0.75:
			sell.append(change[i + 1][j])
		elif forecast[j][0] > 0.6:
			underweight.append(change[i + 1][j])
		elif forecast[j][0] > 0.4:
			middle.append(change[i + 1][j])
		elif forecast[j][0] > 0.25:
			overweight.append(change[i + 1][j])
		else:
			buy.append(change[i + 1][j])
	
	print('buy count:', len(buy), 'overweight count:', len(overweight)
	      , 'middle count:', len(middle), 'underweight count:', len(underweight)
	      , 'sell count:', len(sell))
	
	if not buy:
		buy.append(0)
	if not overweight:
		overweight.append(0)
	if not middle:
		middle.append(0)
	if not underweight:
		underweight.append(0)
	if not sell:
		sell.append(0)
	
	buy = np.mean(np.array(buy))
	overweight = np.mean(np.array(overweight))
	middle = np.mean(np.array(middle))
	underweight = np.mean(np.array(underweight))
	sell = np.mean(np.array(sell))
	
	return buy, overweight, middle, underweight, sell

def svm_train(X, y, change):
	
	for i in range(len(y)):
		svc = SVC(C = 0.8, kernel='rbf', probability=True)
		svc.fit(X[i], y[i])
		current_year = svc.score(X[i],y[i])
		print('current_year: ',2013+i,' score: ',current_year)
		if i+1 < len(y):
			newx_year = svc.score(X[i+1],y[i+1])
			print('newx_year: ', 2014 + i, ' score: ', newx_year)
			forecast = list(svc.predict_proba(X[i+1]))
			
			buy, overweight, middle, underweight, sell = grading(i, forecast, change)
			
			print('Buy predict have change:', buy)
			print('Overweight predict have change:', overweight)
			print('Middle predict have change:', middle)
			print('Underweight predict have change:', underweight)
			print('Sell predict have change:', sell, '\n')

def mlp_grading(i, forecast, change):
	buy = []
	overweight = []
	middle = []
	underweight = []
	sell = []
	# if i == 1:
	# 	print(forecast)
	for j in range(len(forecast)):
		if forecast[j] == -2:
			sell.append(change[i + 1][j])
		elif forecast[j] == -1:
			underweight.append(change[i + 1][j])
		elif forecast[j] == 0:
			middle.append(change[i + 1][j])
		elif forecast[j] == 1:
			overweight.append(change[i + 1][j])
		else:
			buy.append(change[i + 1][j])
	
	print('buy count:', len(buy), 'overweight count:', len(overweight)
	      , 'middle count:', len(middle), 'underweight count:', len(underweight)
	      , 'sell count:', len(sell))
	
	if not buy:
		buy.append(0)
	if not overweight:
		overweight.append(0)
	if not middle:
		middle.append(0)
	if not underweight:
		underweight.append(0)
	if not sell:
		sell.append(0)
	
	buy = np.mean(np.array(buy))
	overweight = np.mean(np.array(overweight))
	middle = np.mean(np.array(middle))
	underweight = np.mean(np.array(underweight))
	sell = np.mean(np.array(sell))
	
	return buy, overweight, middle, underweight, sell
	
def mlp_train(X, y, change):
	for i in range(len(y)):
		mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 5))
		mlp.fit(X[i], y[i])
		current_year = mlp.score(X[i], y[i])
		print('current_year: ', 2013 + i, ' score: ', current_year)
		if i + 1 < len(y):
			newx_year = mlp.score(X[i + 1], y[i + 1])
			print('newx_year: ', 2014 + i, ' score: ', newx_year)
			forecast = list(mlp.predict(X[i + 1]))
			
			buy, overweight, middle, underweight, sell = mlp_grading(i, forecast, change)
			
			print('Buy predict have change:', buy)
			print('Overweight predict have change:', overweight)
			print('Middle predict have change:', middle)
			print('Underweight predict have change:', underweight)
			print('Sell predict have change:', sell, '\n')

if __name__ == '__main__':
	name, X, y, change, mlp_y = loadData(
		['D:\SUSTC\IExperiment\datalearn\\2013全部A股new.xls',
		 'D:\SUSTC\IExperiment\datalearn\\2014全部A股new.xls',
		 'D:\SUSTC\IExperiment\datalearn\\2015全部A股new.xls',
		 'D:\SUSTC\IExperiment\datalearn\\2016全部A股new.xls',
		 'D:\SUSTC\IExperiment\datalearn\\2017全部A股new.xls'])
	
	# X, y, change, mlp_y = loadData(
	# 	['D:\SUSTC\IExperiment\datalearn\\2017全部A股new.xls'])
	#
	# validation_curve(X[4], mlp_y[4])
	
	svm_train(X, y, change)
	print('\n---------------------------------------------------------------\n')
	mlp_train(X, mlp_y, change)
	
	# X_train, X_test, y_train, y_test = ts(X, y, test_size=0.3)
	
	# kernel = 'rbf'
	# clf2016_rbf = SVC(C = 0.9, kernel='poly')
	# clf2016_rbf.fit(X, y)
	# p = list(clf2016_rbf.predict(X))
	# count = 0
	# for i in p:
	# 	if i > 0: count += 1
	# print(count)
	# score_rbf = clf2016_rbf.score(X, y)
	#
	# print("The score of 2016 is : %f" % score_rbf)
	# score_rbf = clf2016_rbf.score(X7, y7)
	# print("The score of 2017 is : %f" % score_rbf)
	# validation_curve()
	# joblib.dump(clf2016_rbf, 'save/clf2016_rbf.pkl')
	# clf3 = joblib.load('save/clf2016_rbf.pkl')
	# print(clf3.score(X,y))