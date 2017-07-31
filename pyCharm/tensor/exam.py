import tensorflow as tf
import matplotlib.pyplot as plt

## 다음 데이터는 익명의 학생들의 모의고사 3회점수와 기말고사 점수이다
## 모의 고사 3회의 점수와 기말고사 점수를 사용해 아직 시험전인 학생의
## 기말고사 결과를 예측하라 ( 리니어 리그리션 )

x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
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

for step in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})

    if step % 10 == 0:
        print(step, "cost:", cost_val, "\nPrediction:\n", hy_val)

testx = [[40., 50., 39.]]
test = sess.run([hypothesis], feed_dict={X: testx})
print("test:", test)