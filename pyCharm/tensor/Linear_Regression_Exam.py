import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

## 선형 회귀 분석 for Tensor Flow
## Example

filename_queue = tf.train.string_input_producer(['Output.csv'], shuffle=False, name='부동산')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

## Date, Rent/buy, Where, Addr, Zipcode, Description, Type, NbOfRooms, Surface, Floor, price, Source
record_defaults = [[""], [""], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

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