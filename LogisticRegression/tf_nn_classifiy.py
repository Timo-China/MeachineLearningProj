import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_regression

def readTrainingData(file_path):
    train_data = np.loadtxt(file_path)
    m, n = train_data.shape
    X = train_data[:, 0: n - 1]
    y = train_data[:,  -1:]
    return X, y

def addLayer(inputX, input_size, output_size, activation_fun = None):
    print(inputX.shape)
    weights = tf.Variable(tf.random_normal([input_size, output_size]))
    bais = tf.Variable(tf.random_normal([1, output_size]))
    h = tf.matmul(inputX, weights) + bais

    if activation_fun == None:
        outputs = h
    else:
        outputs = activation_fun(h)

    return outputs


def trainNNClassify(X,y, learning_rate = 0.01):
    input_X = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='inputsX')
    input_y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='inputsy')

    # 中间加入20个单元的隐藏层
    hidden_layer = addLayer(input_X, 2, 20, activation_fun= tf.nn.softmax)

    # 最后仅有一个输出单元的输出层
    outpus_layer = addLayer(hidden_layer, 20, 2, activation_fun= tf.nn.softmax)

    # 代价函数，用交叉熵 -1/m*sum(y*log(y'))
    cost = tf.reduce_mean(-tf.reduce_sum(input_y * tf.log(outpus_layer), reduction_indices=[1]))

    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init_variable = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_variable)
        for i in range(2000):
            sess.run(train, feed_dict={input_X : X, input_y : y})
            if i % 50 == 0:
                result = sess.run(cost, feed_dict={input_X: X, input_y:y})
                print('cost: %s'% result)
        # 构造测试数据集后预测值, 此处仅预测部分数据
        temp = np.array([[-0.017612, 14.053064],
                        [-1.395634 , 4.662541],
                        [-0.752157, 6.538620],
                        [0.569411 , 9.548755],
                         [1.569411, 0.548755]], dtype= np.float32)
        y_pre = sess.run(outpus_layer, feed_dict={input_X:temp})

        print(sess.run(tf.argmax(y_pre, axis = 1)))




if __name__ == '__main__':
    X,y = readTrainingData('dataset.txt')
    y = np.c_[1-y, y]
    trainNNClassify(X, y, 0.1)









