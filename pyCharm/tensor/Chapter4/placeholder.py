import tensorflow as tf

## x_train = tf.placeholder(tf.float32)
## y_train = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

## 공식
hypothesis = X * W + b

## Cost / Loss Function
cost = tf.reduce_mean(tf.square(hypothesis - Y))  ## 평균 reduce_mean

## optimizer 경사 하강법을 이용한 cost가 제일 낮도록 조정
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
train = optimizer.apply_gradients(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(8001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3, 4, 5], Y: [1.1, 2.3, 3.6, 4.8, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

print("검증:", sess.run(hypothesis, feed_dict={X: [5]}))
