import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def display_partition(x_values, y_values, assignment_values):
    labels = []
    colors = ["red", "blue", "green", "yellow"]
    for i in range(len(assignment_values)):
        labels.append(colors[(assignment_values[i])])
    color = labels
    df = pd.DataFrame(dict(x=x_values, y=y_values, color=labels))
    fig, ax = plt.subplots()
    ax.scatter(df['x'], df['y'], c=df['color'])
    plt.show()

# 군집화 하고자 하는 데이터 개수
num_vectors = 2000

# 군집화할 파티션의 개수
num_clusters = 4

n_samples_per_cluster = 500

# k-means 알고리즘의 계산 반복 횟수
num_steps = 1000

# 초기 입력 데이터 구조체들을 초기화
x_values = []
y_values = []
vector_values = []

# 학습 데이터 집합으로 랜덤한 점들의 집합을 사용
for i in range(num_vectors):
    if np.random.random() > 0.5:
        x_values.append(np.random.normal(0.4, 0.7))
        y_values.append(np.random.normal(0.2, 0.8))
    else:
        x_values.append(np.random.normal(0.6, 0.4))
        y_values.append(np.random.normal(0.8, 0.5))


# 파이썬의 zip 함수를 사용해서 vector_values의 전체 목록을 얻음
vector_values = []
for x, y in zip(x_values, y_values):
    vector_values.append([x, y])

# vector_values를 텐서플로에서 사용 가능한 상수로 변환
vectors = tf.constant(vector_values)

# 작성한 학습 데이터 집합 확인
plt.plot(x_values, y_values, 'o', label="Input Data")
plt.legend()
#plt.show()

# 4개의 센트로이드를 만들고 tf.random_shuffle을 사용해 인덱스를 지정
n_samples = tf.shape(vector_values)[0]  # [2000 2] > 2000
random_indices = tf.random_shuffle(tf.range(0, n_samples))  # 엘리먼트를 섞음

begin = [0, ]
size = [num_clusters, ]     # 4,
size[0] = num_clusters

# 각 인덱스들에 대한 초기 센트로이드를 가짐
centroid_indices = tf.slice(random_indices, begin, size)
centroids = tf.Variable(tf.gather(vector_values, centroid_indices))

# 비용 함수와 최적화
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

# 두 텐서의 형태를 표준화
vectors_subtration = tf.subtract(expanded_vectors, expanded_centroids)

# euclidean_distances 비용 함수 작성
euclidean_distances = tf.reduce_sum(tf.square(vectors_subtration), 2)
assignments = tf.to_int32(tf.argmin(euclidean_distances, 0))

partitions = [0, 0, 1, 1, 0]
num_partitions = 2
data = [10, 20, 30, 40, 50]
#outputs[0] = [10, 20, 50]      ?
#outputs[1] = [30, 40]          ?

# 각 샘플에 대해 가장 가까운 인덱스를 찾고 dynamic_partition 함수를 사용해 새로운 그룹으로 묶음
partitions = tf.dynamic_partition(vectors, assignments, num_clusters)

# 하나의 그룹에 대해 tf.reduce_mean을 실행해 군집의 평균을 찾고 새로운 센트로이드로 지정
update_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)

# 테스트와 알고리즘 평가
#       모든 변수를 초기화하고 평가 그래프를 인스턴스화
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

# 계산 시작
for step in range(num_steps):
    _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

# 결과 출력 함수 호출
display_partition(x_values, y_values, assignment_values)

# 시각화
plt.plot(x_values, y_values, 'o', label='Input Data')
plt.legend()
plt.show()

