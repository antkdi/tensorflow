import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

## 모의 고사 3회의 점수와 기말고사 점수를 사용해 시험전인 학생의 기말고사 결과를 예측하라 ( 리니어 리그리션 )

## 파일 로드 ## 텐서에서 제공하는 파일 큐 - 여러개의 파일을 배치로 가져다 쓸수 있다.
filename_queue = tf.train.string_input_producer(['data-01-test-score.csv', 'data-02-test-score.csv'], shuffle=False, name='scoreCard')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

## 레코드의 디폴트 타입 정의
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

## 배치 데이터의 의 규격 정의
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

## 입력피처의 텐서
X = tf.placeholder(tf.float32, shape=[None, 3])
## 출력 텐서
Y = tf.placeholder(tf.float32, shape=[None, 1])

## 가중치
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
## 바이어스
b = tf.Variable(tf.random_normal([1]), name='bias')

##
hypothesis = tf.matmul(X, W) + b

## cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))

## opt
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "cost:", cost_val, "\n,Prediction: \n", hy_val)


print("Training End..\n\n\n\n")
## ask my Score
print("Your score will be ", sess.run([hypothesis], feed_dict={X: [[30, 20, 40]]}))
print("\n")
print("My Friends Scores will be \n", sess.run([hypothesis], feed_dict={X: [[50, 44, 78], [92, 88, 99]]}))