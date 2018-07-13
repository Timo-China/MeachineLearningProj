from __future__ import print_function, division
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

def LoadData(data_file):
    data = pd.read_csv(data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]].values,
        data["Occupancy"].values.reshape(-1, 1), random_state=42)

    # one-hot 编码
    y_train = tf.concat([1 - y_train, y_train], 1)
    y_test = tf.concat([1 - y_test, y_test], 1)

    return X_train, X_test, y_train, y_test

def Training(X_train, X_test, y_train, y_test):
    learnning_rate = 0.001
    training_num = 50
    batch_size = 50

    n_samples = X_train.shape[0]
    n_features = 5
    n_class = 2
    display_step = 1
    x = tf.placeholder(tf.float32, shape=[None, 5]) # 5个特征
    y = tf.placeholder(tf.float32, shape=[None, 2]) # 2个输出，0和1
    W = tf.Variable(tf.zeros([5, 2]), name='Weight')
    b = tf.Variable(tf.zeros([2]), name='bais')
    pre = tf.matmul(x, W) + b
    # J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pre, y))
    J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pre, labels=y))
    g = tf.train.GradientDescentOptimizer(learning_rate = learnning_rate).minimize(J)

    correct_pre = tf.equal(tf.argmax(pre, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_num):
            avg_cost = 0
            total_batch = int(n_samples / batch_size)
            for i in range(total_batch):
                _, c = sess.run([g, J],
                                feed_dict={x: X_train[i * batch_size: (i + 1) * batch_size],
                                           y: y_train[i * batch_size: (i + 1) * batch_size, :].eval()})
                avg_cost = c / total_batch
            plt.plot(epoch + 1, avg_cost, 'co')

            if (epoch + 1) % display_step == 0:
                print("Epoch:", "%04d" % (epoch + 1), "cost=", avg_cost)
            validate_acc = sess.run(accuracy, feed_dict={x: X_test,
                                                         y: y_test.eval()})
            print('Test Accuracy:', validate_acc)

        print("Optimization Finished!")

        print("Testing Accuracy:", accuracy.eval({x: X_train, y: y_train.eval()}))
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = LoadData('occupancy_data/datatraining.txt')
    Training(X_train, X_test, y_train, y_test)

