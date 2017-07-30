import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

# 데이터 읽기
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 학습 및 테스트 데이터 읽기
train_pixels,train_list_values = mnist.train.next_batch(100)
test_pixels,test_list_of_values  = mnist.test.next_batch(10)

# 분류기를 제작하는데 사용할 텐서 정의
train_pixel_tensor = tf.placeholder("float", [None, 784])
test_pixel_tensor = tf.placeholder("float", [784])

# 비용 함수 설정 - 픽셀간의 거리
#       reduce 함수는 텐서의 차원들을 탐색하며 개체들의 총합을 계산
distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor, tf.negative(test_pixel_tensor))), reduction_indices=1)
# 거리 함수를 최소화하기 위해 arg_min을 사용, 가장 작은 거리를 갖는 인덱스(최근접 이웃)를 리턴
pred = tf.arg_min(distance, 0)  # tf.arg_min(input, demension)

# 테스트와 알고리즘 평가

# 정확도 및 변수 초기화
accuracy = 0.
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(test_list_of_values)):
        # pred 함수를 사용해 최근접 이웃 인덱스를 평가
        nn_index = sess.run(pred, feed_dict={train_pixel_tensor:train_pixels, test_pixel_tensor:test_pixels[i, :]})
        print("nn_index", nn_index)
        # 최근접 이웃의 클래스 레이블 확인 및 실제 레이블과 비교
        print("Test N ", i, "Predicted Class: ", np.argmax(train_list_values[nn_index]), "True Class: ", np.argmax(test_list_of_values[i]))
        # 분류기에 대한 정확도 평가
        if np.argmax(train_list_values[nn_index]) == np.argmax(test_list_of_values[i]):
            accuracy += 1./len(test_pixels)
        # 오답에 대한 그림 출력
        else:
            image = test_pixels[i, :]
            image = np.reshape(image, [28, 28])
            plt.imshow(image)
            plt.show()
    # 분류기의 정확도 출력
    print("Result = ", accuracy)

