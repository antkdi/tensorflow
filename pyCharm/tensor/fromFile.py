import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

## 모의 고사 3회의 점수와 기말고사 점수를 사용해 시험전인 학생의 기말고사 결과를 예측하라 ( 리니어 리그리션 )

input_data = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

x_data = input_data[:, 0:-1]
y_data = input_data[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data, len(y_data))

y_data = [[152.], [185.], [180.], [196.], [142.]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

## cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))

## opt
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(20001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})

    if step % 10 == 0:
        print(step, "cost:", cost_val, "\n,Prediction: \n", hy_val)

print("Training End..\n\n\n\n")
## ask my Score
print("Your score will be ", sess.run([hypothesis], feed_dict={X: [[30, 20, 40]]}))
print("\n")
print("My Friends Scores will be \n", sess.run([hypothesis], feed_dict={X: [[50, 44, 78], [92, 88, 99]]}))