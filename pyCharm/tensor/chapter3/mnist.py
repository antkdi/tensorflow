import input_data
import numpy as np
import matplotlib.pyplot as plt

# input_data를 이용해 데이터 집합을 읽음
mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=False)

# 첫 10개 이미지에 대해서 픽셀 행렬과 픽셀에 해당하는 숫자를 읽음
pixels, real_values = mnist_images.train.next_batch(10)

print("list of values loaded ", real_values)
example_to_visualize = 1    # 1번째 그림
print("element N " + str(example_to_visualize + 1) + " of the list plotted")

# 픽셀 행렬을 이용하여 그림 표시
image = pixels[example_to_visualize,:]
image = np.reshape(image, [28,28])
plt.imshow(image)
plt.show()