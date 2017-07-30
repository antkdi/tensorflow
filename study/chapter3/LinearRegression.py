import numpy as np
import matplotlib.pyplot as plt #pip install matplotlib
import tensorflow as tf

# 1. 알고리즘을 테스트하기 위해 2차원 공간상의 데이터 생성

# 생성할 점의 개수
number_of_points = 500

# 리스트 초기화
x_point = []
y_point = []

# y와 x의 선형 관계를 정의하는 A와 b 두 상수 값을 지정
a = 0.22
b = 0.78

# numpy를 이용한 정규분포 값 생성
for i in range(number_of_points):
    x = np.random.normal(0.0, 0.5)
    y = a * x + b + np.random.normal(0.0, 0.1)
    x_point.append([x])
    y_point.append([y])

# graph 출력
plt.plot(x_point, y_point, 'x', label='Input Data')     # x_data, y_data, mark, label
plt.legend()
plt.show()

# 2. 모델 생성
#       목표 : 모델을 생성하여 입력 데이터 x에 대해 y를 예측
#       선형 회귀 알고리즘은 두개의 상수 A와 b에 의해 결정
#       이 정보를 모른다고 가정하고 모델을 만들어 예측

# A와 b를 tf.Variable로 정의하고, 임의의 값을 할당
A = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # random_uniform(shape, min, max)
b = tf.Variable(tf.zeros([1]))      # zeros(shape)

# 변수 y와 변수 x의 선형 관계 수식을 정의
y = A * x_point + b

# 비용 함수 정의 - 평균 제곱근 오차(mean square error)를 사용
cost_function = tf.reduce_mean(tf.square(y - y_point))

# 최적화 함수 정의 - 경사 하강법(Gradient Descent)을 사용
optimizer = tf.train.GradientDescentOptimizer(0.5)   # GradientDescentOptimizer(learning rate)

# 3. 모델 학습

# 변수 초기화
train = optimizer.minimize(cost_function)
model = tf.global_variables_initializer()

# 학습
with tf.Session() as session:
    session.run(model)
    for step in range(0, 20):
        session.run(train)
        if(step % 5) == 0:
            plt.plot(x_point, y_point, 'o', label='step = {}'.format(step))
            plt.plot(x_point, session.run(A) * x_point + session.run(b))
            plt.legend()
            plt.show()

    # 정답 : 0.22, 0.78
    # 결과 값 : [ 0.22260258] [ 0.78171444]
    print(session.run(A),session.run(b))
