import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def readTrainingData(file_path):
    train_data = np.loadtxt(file_path)
    m, n = train_data.shape
    X = train_data[:, 0: n - 1]
    y = train_data[:,  -1:]
    return X, y


def trainData(X, y, learn_rate = 0.001):
    m, n = X.shape;
    #y = np.hstack((1 - y, y))
    temp = np.ones([m, 1])
    X = np.c_[temp, X]

    m, n = X.shape;
    input_X = tf.placeholder(shape=[None, 3],dtype=tf.float32, name='inputX')
    input_y = tf.placeholder(shape=[None, 1],dtype=tf.float32, name='inputY')
    weights = tf.Variable(initial_value=tf.ones([3, 1]))
    h = tf.sigmoid(tf.matmul(input_X, weights))

    cost = -tf.reduce_mean(input_y * tf.log(h) + (1-input_y)*tf.log(1 - h))
    train = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

    init_value = tf.global_variables_initializer()
    w = []
    with tf.Session() as sess:
        sess.run(init_value)
        for i in range(5000):
            sess.run(train, feed_dict={input_X:X, input_y:y})
            c = sess.run(cost, feed_dict={input_X:X, input_y:y})
            print(c)
        w = sess.run(weights, feed_dict={input_X:X, input_y:y})
    return w

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
    X,y = readTrainingData('dataset.txt')
    w = trainData(X, y, 0.3)
    plotTrainData(X,y, w)
