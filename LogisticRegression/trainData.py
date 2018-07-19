
import numpy as np
import matplotlib.pyplot as plt

def readTrainingData(file_path):
    train_data = np.loadtxt(file_path)
    m, n = train_data.shape
    X = train_data[:, 0: n - 1]
    y = train_data[:,  -1: ]
    return X, y

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
'''
梯度下降
'''

def gradentDescent(X, y):
    m, n = X.shape;
    temp = np.ones([m, 1])
    X = np.c_[temp, X] # 在X前加入全1列
    alpha = 0.01
    max_cycles = 1000

    weight = np.random.random((n + 1, 1))

    for i in range(max_cycles):

        g = sigmoid(np.dot(X , weight))
        error = g - y
        print(error.shape)
        print(np.dot(X.transpose(), error).shape)
        weight = weight - alpha * np.dot(X.transpose() , error)

    return weight

def plotTrainData(X, y, weight):
    fig = plt.figure(1)

    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for i in range(y.shape[0]):
        if y[i] == 0:
            x0.append(X[i, 0])
            y0.append(X[i, 1])
        else:
            x1.append(X[i, 0])
            y1.append(X[i, 1])

    ax = fig.add_subplot(1,1,1)
    ax.scatter(x0, y0, s = 20, c = 'r', marker='+')
    ax.scatter(x1, y1, s = 20, c='b', marker='*')
    x = np.arange(-3.0, 3.0, 0.1)

    y = (-weight[0]-weight[1]*x)/weight[2]

    ax.plot(x, y)
    plt.xlabel('X');
    plt.ylabel('Y')
    plt.show()

if __name__ == '__main__':
    X, y = readTrainingData('dataset.txt')
    weight = gradentDescent(X, y)
    plotTrainData(X, y, weight)
