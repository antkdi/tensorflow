import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


## 로지스틱 회귀 분석


##input_data = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)

##x_data = input_data[:, 0:-1]
##y_data = input_data[:, [-1]]

##print(x_data.shape, x_data, len(x_data))
##print(y_data.shape, y_data, len(y_data))


x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

## Placeholder
## 입력값 Shape
X = tf.placeholder(tf.float32, shape=[None, 4])
## 출력값 Shape
Y = tf.placeholder(tf.float32, shape=[None, 3])

nb_classes = 3

## 가중치, 편향 변수 Shape가 맞아야함 분류용 입력 값이 4개이고 분류할 클래스가 3개이므로 가중치의 Shape 는 4, 3
W = tf.Variable(tf.random_normal([4, nb_classes], name='weight'))
b = tf.Variable(tf.random_normal([nb_classes], name='bias'))

## 가설 소프트맥스 ( 예측 값을 확률 분류 확률로 계산 )
## softmax = exp(logits) / reduce_sum(exp(logits),dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

## Cross entropy  Cost/ Loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis= 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

##Launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={ X: x_data, Y: y_data})
        if step % 200 == 0:
            cost_val, hy_val = sess.run([cost, hypothesis], feed_dict={X: x_data, Y: y_data})



    myHy = sess.run(hypothesis, feed_dict={X: [[1, 2, 1, 2]]})
    mysoft = sess.run(tf.arg_max(myHy, 1))
    print("myData :", "\n", mysoft)
