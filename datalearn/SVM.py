# Author:ZZR


from sklearn.svm import SVC
import numpy as np

x_train = np.array([
                    [4, 5, 6],
                    [3, 5, 3],
                    [1, 7, 2],
                    [1, 2, 3],
                    [1, 3, 4],
                    [2, 1, 2]])

y_train = np.array([1, 1, 1, -1, -1, -1])

x_test = np.array([[2, 2, 2],
                   [3, 2, 6],
                   [1, 7, 4]])

clf = SVC(probability=True)
clf.fit(x_train, y_train)

# 返回预测标签
print(clf.predict(x_test))

# 返回预测属于某标签的概率
print(clf.predict_proba(x_test))