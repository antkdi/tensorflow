Logistic_Sigmoid.ipynb                                                                              000644  000766  000024  00000043163 13137603537 017262  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 로지스틱 회귀\n",
    "> 일반적인 선형회귀로 풀수 없는 지도학습 중 하나인 '분류' 문제를 해결하기 위한 알고리즘\n",
    "> 활성화 함수로 시그모이드 함수를 사용한다.\n",
    "> "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting ../mnist/train-images-idx3-ubyte.gz\n",
      "Extracting ../mnist/train-labels-idx1-ubyte.gz\n",
      "Extracting ../mnist/t10k-images-idx3-ubyte.gz\n",
      "Extracting ../mnist/t10k-labels-idx1-ubyte.gz\n",
      "WARNING:tensorflow:From <ipython-input-14-1f2ea0b3c55d>:58: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.\n",
      "Instructions for updating:\n",
      "Use `tf.global_variables_initializer` instead.\n",
      "Epoch: 0001 cost= 1.174406662\n",
      "Epoch: 0002 cost= 0.661961771\n",
      "Epoch: 0003 cost= 0.550479427\n",
      "Epoch: 0004 cost= 0.496727534\n",
      "Epoch: 0005 cost= 0.463699739\n",
      "Epoch: 0006 cost= 0.440824241\n",
      "Epoch: 0007 cost= 0.423898390\n",
      "Epoch: 0008 cost= 0.410591180\n",
      "Epoch: 0009 cost= 0.399882070\n",
      "Epoch: 0010 cost= 0.390926550\n",
      "Epoch: 0011 cost= 0.383350126\n",
      "Epoch: 0012 cost= 0.376758833\n",
      "Epoch: 0013 cost= 0.371010472\n",
      "Epoch: 0014 cost= 0.365936794\n",
      "Epoch: 0015 cost= 0.361384122\n",
      "Epoch: 0016 cost= 0.357272247\n",
      "Epoch: 0017 cost= 0.353554923\n",
      "Epoch: 0018 cost= 0.350088185\n",
      "Epoch: 0019 cost= 0.347031046\n",
      "Epoch: 0020 cost= 0.344179152\n",
      "Epoch: 0021 cost= 0.341420337\n",
      "Epoch: 0022 cost= 0.338996688\n",
      "Epoch: 0023 cost= 0.336667450\n",
      "Epoch: 0024 cost= 0.334440362\n",
      "Epoch: 0025 cost= 0.332434435\n",
      "Training phase finished\n"
     ]
    },
    {
     "data": {
      "image/png": "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\nBQUFvn37du/UqZPv3bvXDx065P369Quuq6Jbb73Ve/fu7cXFxf7ZZ595+/bt/Z///KcvW7bM09LS\nvKCgwEtLSz0rK8tXrlx51HUcOXLEr7rqqqD2M844ww8dOnTU9zk3N9efe+45d3ffu3evd+nSxYuL\ni4+po6qfKyDPa/E7NulbClA2y+iv0wbxyV3D+eu0QQ2iu0GiF8vxoZUrVzJmzBgArrnmGv7yl78E\n26+88kqAYH9l/fr149e//jV33303n376KampNXdVLl++nB/96EdBl0jr1q2rPC43N5euXbsyZswY\npk6dCsDmzZv54IMPGDJkCJmZmfzqV78iPz+fwsJC9u/fT79+/aqsdciQIcHzLF26lKVLl9K7d2/O\nP/98/v73v7NlyxZ69uzJsmXLmDp1Km+99RatWrWiVatWNG/enBtuuIEFCxZw8sknH3Xeffv2UVhY\nyIABAwAYN24cb775ZrD/iiuuAKBPnz5s27atxu9LZf369aNjx47B4zlz5gStgfz8fD766CMAUlNT\nGTZs2HGfp7paLrnkEk499VRSU1P55je/yfbt23n33XcZNGgQp512GieddFKN40GjRo2iefPmfP3r\nX+fb3/42q1atAqBv3760a9eOpk2bkpmZGTznihUryM7OplevXrzxxhts2LABgO7duzN27FjmzZsX\n3Ii2dOlS7rzzTjIzM/nOd77DwYMHg1ZsfWkUoSDJobpxoIY2PjRmzBgWL15Mamoql156KX/+85/r\n5byzZ8/mww8/5O677+b6668Hylr63bt3Z926daxbt47169ezdOnS456rRYsWwefuzvTp04NzbN26\nlRtuuIGuXbuyZs2aoKvsjjvuoFmzZvztb39j9OjRvPzyywwdWtOal8f62te+BkDTpk0pLS2N6msr\n1rx8+XLefPNN3nnnHd577z0yMjKCu+JPOumk4Lianqe6Wsq317XOylNCyx9Xdd6ioiImT57MwoUL\nef/997n++uuD61iyZAmTJk1i1apVZGdnc/jwYdydRYsWBa/V9u3b6dq1a1T1HY9CQRJGLMeHLrro\nIubPnw/AvHnz+Na3vgWU/bX3xz/+ESDYX9nHH3/MOeecw0033cTIkSN5//33admyJfv376/y+CFD\nhvDwww8Hv3z27t1bY22TJ0/myJEjLFmyhPT0dHbv3s3KlSuBsmVDNmzYQFpaGi1btuTdd9+tsVaA\n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\n/H9Sk/MT/POZuAAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x120d556a0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model accuracy: 0.9136\n"
     ]
    }
   ],
   "source": [
    "import input_data\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "mnist = input_data.read_data_sets(\"../mnist/\", one_hot=True)\n",
    "\n",
    "\n",
    "## 매개변수 생성\n",
    "training_epochs = 25\n",
    "\n",
    "learning_rate = 0.01\n",
    "batch_size = 100\n",
    "display_step =1\n",
    "\n",
    "# 모델 생성\n",
    "\n",
    "# 픽셀 데이터를 저장할 텐서 선언 28 * 28 의 크기를 갖는 이미지\n",
    "x = tf.placeholder(\"float\",[None,784])\n",
    "\n",
    "# 10개의 확률 값을 저장할 텐서 선언  - 모든 확률 값의 합은 1이 되어야 한다\n",
    "y = tf.placeholder(\"float\",[None,10])\n",
    "\n",
    "# 각 이미지에 확률값을 구하기 위해 활성함수 사용 ( 소프트맥스 )\n",
    "# 소프트맥스는 특정이미지에 대한 근거를 계산, 근거를 10가지 후보 클래스에 해당하는 확률로 변환\n",
    "# 28*28 인 이미지가 10개인 변수\n",
    "# 근거를 평가 하기 위해 텐서 가중치를 선언 \n",
    "W = tf.Variable(tf.zeros([784,10]))\n",
    "\n",
    "# 주어진 이미지에 대해 각 클래스마다 입력 텐서 x 에 W를 곱함으로 근거를 평가 ( 가중치 )\n",
    "#evidence = tf.matmul(x,W)\n",
    "\n",
    "# 근거를 정의하기 위해 편향 텐서를 정의\n",
    "b = tf.Variable(tf.zeros([10]))\n",
    "\n",
    "# 일반적으로 모델은 불확실성 정도를 의미하는 편향에 대한 별도의 매개변수를 갖는다 ( 편향 - 바이어스)\n",
    "evidence = tf.matmul(x,W) + b\n",
    "\n",
    "\n",
    "\n",
    "# 확률 값을 갖는 출력벡터를 얻기위해 소프트 맥스를 사용 ( 활성함수라 명명)\n",
    "activation = tf.nn.softmax(tf.matmul(x,W)+b)\n",
    "\n",
    "# 모델의 성능 정도를 판단할 방법을 정해야한다\n",
    "# 목표는 모델의 나쁜 정도를 나타내는 매티륵 값이 최소가 되는 매개변수 W와 b의 값을 얻고자 하는것이다.\n",
    "\n",
    "# cross-entropy error 함수를 사용할 것이다.\n",
    "cross_entropy = y * tf.log(activation)\n",
    "\n",
    "#Cost\n",
    "cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices=1))\n",
    "\n",
    "#경사 하강법을 이용해 비용 최소하\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)\n",
    "\n",
    "avg_set =[]\n",
    "epoch_set =[]\n",
    "\n",
    "init = tf.initialize_all_variables()\n",
    "\n",
    "## 세션 시작\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "    \n",
    "    #학습주기\n",
    "    for epoch in range(training_epochs):\n",
    "        avg_cost =0.\n",
    "        total_batch = int(mnist.train.num_examples/batch_size)\n",
    "        \n",
    "        #모든 배치에 대해 반복\n",
    "        for i in range(total_batch):\n",
    "            batch_xs,batch_ys = mnist.train.next_batch(batch_size)\n",
    "            sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})\n",
    "            avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys})/total_batch\n",
    "            \n",
    "        # 각 반복 단계마다 로그 출력\n",
    "        if epoch % display_step ==0:\n",
    "            print (\"Epoch:\", '%04d' % (epoch+1), \"cost=\", \"{:.9f}\".format(avg_cost))\n",
    "    \n",
    "        avg_set.append(avg_cost)\n",
    "        epoch_set.append(epoch+1)\n",
    "                \n",
    "        \n",
    "    print (\"Training phase finished\")\n",
    "    plt.plot(epoch_set,avg_set, 'o', label='Logistic Regression Traninng phase')\n",
    "    plt.ylabel('cost')\n",
    "    plt.xlabel('epoch')\n",
    "    plt.legend()\n",
    "    plt.show()\n",
    "        \n",
    "    correct_prediction = tf.equal(tf.argmax(activation,1),tf.argmax(y,1))\n",
    "        \n",
    "    accuracy = tf.reduce_mean(tf.cast(correct_prediction,\"float\"))\n",
    "    print (\"Model accuracy:\", accuracy.eval({ x:mnist.test.images, y:mnist.test.labels}))\n",
    "        \n",
    "        \n",
    "                \n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 실전 예제\n",
    "#### "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
                                                                                                                                                                                                                                                                                                                                                                                                             MLP.ipynb                                                                                           000644  000766  000024  00000042176 13137603537 014465  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 다중 계층 퍼셉트론 (MLP)\n",
    "### MLP 네트워크에 대한 일반적인 학습 알고리즘은  '역전파 알고리즘' 을 사용한다.\n",
    "##### 시스템의 결과값과 기대값을 비교한다. 계산된 차이 ( 즉, 에러)에 기초에 이 알고리즘은 신경망의 시냅스에 해당하는 가중치를 수정"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting ../mnist/train-images-idx3-ubyte.gz\n",
      "Extracting ../mnist/train-labels-idx1-ubyte.gz\n",
      "Extracting ../mnist/t10k-images-idx3-ubyte.gz\n",
      "Extracting ../mnist/t10k-labels-idx1-ubyte.gz\n",
      "Epoch 0001 cost= 2.116159168\n",
      "Epoch 0002 cost= 0.597048687\n",
      "Epoch 0003 cost= 0.398541482\n",
      "Epoch 0004 cost= 0.292631849\n",
      "Epoch 0005 cost= 0.223589640\n",
      "Epoch 0006 cost= 0.173506536\n",
      "Epoch 0007 cost= 0.136102995\n",
      "Epoch 0008 cost= 0.106170934\n",
      "Epoch 0009 cost= 0.084325237\n",
      "Epoch 0010 cost= 0.066066858\n",
      "Epoch 0011 cost= 0.052297312\n",
      "Epoch 0012 cost= 0.040624121\n",
      "Epoch 0013 cost= 0.031596599\n",
      "Epoch 0014 cost= 0.024813710\n",
      "Epoch 0015 cost= 0.019010525\n",
      "Epoch 0016 cost= 0.014793412\n",
      "Epoch 0017 cost= 0.011264293\n",
      "Epoch 0018 cost= 0.008618943\n",
      "Epoch 0019 cost= 0.006610691\n",
      "Epoch 0020 cost= 0.005329992\n",
      "Traning phase finished\n"
     ]
    },
    {
     "data": {
      "image/png": "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\nETcBHwRGR0QbMBcYDpCZ1wGnAZ+LiG3AZuD0zMyy6pEk9ay0UMjMM3qYfy2VQ1YlSf1Es48+kiT1\nI4aCJKlgKEiSCoaCJKlgKEiSCoaCJKlgKEiSCoaCJKlgKEiSCoaCJKlgKEiSCoaCJKlgKEiSCoaC\nJKlgKEiSCoaCJKlgKEiSCoaCJKlgKEiSCoaCJKlgKEiSCoaCJKlgKEiSCoaCJKlgKEiSCoaCJKlg\nKEiSCoaCJKlQUyhExOxapkmSBrZa9xQurXGaJGkA27W7mRFxMjADGBMR32g36+3AtjILkyQ1Xreh\nADwHLANOAR5tN30T8D/KKkqS1BzdhkJmrgJWRcT3M3MrQETsBRyQmb9rRIGSpMapdUzh7oh4e0Ts\nDSwH/jkirimxLklSE9QaCntm5u+BU4EbM/MY4MPllSVJaoZaQ2HXiNgP+ARwR4n1SJKaqNZQuAJY\nAvw2M5dGxMHAb8orS5LUDD0dfQRAZt4K3Nru9VPAX5ZVlCSpOWo9o3lsRCyMiJeqj9siYmwPy9xQ\nbftYF/MjIr4REWsjYnVEvG9nfgBJUt+ptfvoX4HFwP7Vx4+q07qzADipm/knA4dVH+cC/1RjLZKk\nktQaCi2Z+a+Zua36WAC0dLdAZv4ceKWbJjOpHMmUmfkwMKo6mC1JapJaQ2FDRMyJiGHVxxxgQ52f\nPQZ4tt3rtuo0SVKT1BoK51A5HPUF4HngNODskmp6i4g4NyKWRcSy9evXN+pjJWnI6c0hqWdlZktm\nvoNKSHy5zs9eBxzQ7vXY6rS3yMzrM7M1M1tbWrrttZIk1aHWUJjY/lpHmfkKMLnOz14MnFk9Cmka\n8GpmPl/ne0qS6lDTeQrALhGx1/ZgqF4DqafLbt8EfBAYHRFtwFxgOEBmXgfcSeWy3GuB14HP7MwP\nIEnqO7WGwt8D/ycitp/ANhv4u+4WyMwzepifwOdr/HxJUgPUekbzjRGxDDihOunUzHy8vLIkSc1Q\n654C1RAwCCRpEKt1oFmSNAQYCpKkgqEgSSoYCpKkgqEgSSoYCpKkgqEgSSoYCpKkgqEgSSoYCpKk\ngqEgSSoYCpKkgqEgSSoYCpKkgqEgSSoYCpKkgqEgSSoYCpKkgqEgSSoYCpKkgqEgSSoYCpKkgqEg\nSSoYCpKkgqEgSSoYCpKkgqEgSSoYCpKkwq7NLmAgWLRiHVcveZLnNm5m/1EjuXj6eGZNHtPssiSp\n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\n400AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x118e47dd8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Accuracy: 0.9437\n"
     ]
    }
   ],
   "source": [
    "import input_data\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "mnist = input_data.read_data_sets(\"../mnist\", one_hot=True)\n",
    "\n",
    "## 신경망에 대한 학습률 \n",
    "learning_rate = 0.001\n",
    "\n",
    "## 반복 횟수\n",
    "training_epochs = 20\n",
    "\n",
    "## 배치 한번에 분류할 이미지 수\n",
    "batch_size = 100\n",
    "display_step = 1\n",
    "\n",
    "## 첫번째 계층의 뉴런 수\n",
    "n_hidden_1 = 256\n",
    "\n",
    "## 두번째 계층의 뉴런 수\n",
    "n_hidden_2 = 256\n",
    "\n",
    "\n",
    "## 입력값의 크기\n",
    "n_input = 784  ## 28*28\n",
    "\n",
    "## 출력 클래스의 크기\n",
    "n_classes = 10\n",
    "\n",
    "## 입력 및출력의 크기를 완벽히 정의했고, 은닉 계층의 개수와 각 계층에서의 뉴런 개수를 정하는 방법에 대한 엄격한 기준은 없다.\n",
    "\n",
    "\n",
    "### 모델생성\n",
    "\n",
    "# 입력 텐서\n",
    "x = tf.placeholder(\"float\",[None,n_input])\n",
    "\n",
    "## 출력 텐서\n",
    "y = tf.placeholder(\"float\",[None,n_classes])\n",
    "\n",
    "# 각 계층의 노드 수\n",
    "h = tf.Variable(tf.random_normal([n_input, n_hidden_1]))\n",
    "\n",
    "# 계층 1에대한 편향\n",
    "bias_layer_1 = tf.Variable(tf.random_normal([n_hidden_1]))\n",
    "\n",
    "## 레이어 1은 내적곱 + 편향 에 대한 결과를 전달\n",
    "layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,h),bias_layer_1))\n",
    "\n",
    "## 결과값을 활성화 함수를 통해 다음 계층의 뉴런으로 전달한다\n",
    "## 은닉계층에 속한 뉴런의 활성화 함수는 선형이 될수 없다!?\n",
    "## 두번째 중간 계층 256 * 256 \n",
    "w = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))\n",
    "\n",
    "## 두번째 계층의 편향 텐서\n",
    "bias_layer_2 = tf.Variable(tf.random_normal([n_hidden_2]))\n",
    "\n",
    "## 두번째 계층의 뉴런은 계층1의 뉴런으로부터 입력값을 전달 받고 가중치 연결과 결합 후 계층2편향치를 더한다\n",
    "layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,w),bias_layer_2))\n",
    "\n",
    "## 결과 값은 다음 계층인 출력 계층으로 전달된다.\n",
    "output = tf.Variable(tf.random_normal([n_hidden_2, n_classes]))\n",
    "bias_output = tf.Variable(tf.random_normal([n_classes]))\n",
    "output_layer = tf.matmul(layer_2, output) + bias_output\n",
    "\n",
    "## 출력계층은 두번째 계층으로 부터 256 개의 입력 신호를 받게 되는데 이 값은 각 숫자에 대한 클래스에 속할 확률로 변환된다\n",
    "## 로지스틱 회귀를 위해 비용함수를 정의 한다.\n",
    "## tf.nn.softmax_cross_entropy_with_logits 함수는 소프트 맥스 계층에 대한 비용을 계산 - 로짓은 모델의 출력값으로 정규화 되지않은 로그 확률\n",
    "#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer,y))\n",
    "cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer,labels=y))\n",
    "\n",
    "## 비용함수를 최소화할 옵티마이저는 다음과 같다\n",
    "optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)\n",
    "\n",
    "##세션 실행\n",
    "\n",
    "## 그래프에 사용할 설정을 정의\n",
    "avg_set = []\n",
    "epoch_set =[]\n",
    "\n",
    "#변수 초기화\n",
    "init = tf.global_variables_initializer()\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "    \n",
    "    ## 학습 반복 횟수\n",
    "    for epoch in range(training_epochs):\n",
    "        avg_cost = 0.\n",
    "        total_batch = int(mnist.train.num_examples/batch_size)\n",
    "        \n",
    "        ## 배치에 대해  total_batch 만큼 반복\n",
    "        for i in range(total_batch):\n",
    "            ## batch_size 만큼 학습 데이터를 가져옴 ( 100개 )\n",
    "            batch_xs,batch_ys = mnist.train.next_batch(batch_size)\n",
    "            ## 학습수행 옵티마이저와 100개 학습 데이터를 인자로 \n",
    "            sess.run(optimizer, feed_dict={x: batch_xs, y:batch_ys})\n",
    "            ## 학습된 내용에 대한 비용 값을 토탈로 나눠서 평균 코스트 계산 \n",
    "            avg_cost += sess.run(cost,feed_dict={x:batch_xs, y:batch_ys})/total_batch\n",
    "            \n",
    "        if(epoch % display_step) == 0:\n",
    "            print(\"Epoch\",'%04d' % (epoch +1),\"cost=\",\"{:.9f}\".format(avg_cost))\n",
    "        avg_set.append(avg_cost)\n",
    "        epoch_set.append(epoch+1)\n",
    "    print(\"Traning phase finished\")\n",
    "    \n",
    "    ## 에폭 셋, 평균셋에 대한 시각화\n",
    "    plt.plot(epoch_set, avg_set,'o',label='MLP Traning phase')\n",
    "    plt.ylabel('cost')\n",
    "    plt.xlabel('epoch')\n",
    "    plt.legend()\n",
    "    plt.show();\n",
    "    \n",
    "    ## 모델평가 arg_max를 씀 ( 가장 높은값의 index를 반환 ) \n",
    "    ## output_layer의 classes는 10개이고 10개에 대한 확률에 대해 가장 높은값을 리턴하고 \n",
    "    correct_prediction = tf.equal(tf.argmax(output_layer,1),tf.argmax(y,1))\n",
    "    \n",
    "    ## 정확도 평가\n",
    "    accuracy = tf.reduce_mean(tf.cast(correct_prediction,\"float\"))\n",
    "    print(\"Model Accuracy:\", accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
                                                                                                                                                                                                                                                                                                                                                                                                  Output.csv                                                                                          000644  000766  000024  00000355424 13137603537 015012  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         ,Date,Rent/Buy,Where,Address,ZipCode,Description,Type,NbOfRooms,Surface,Floor,Price,Source
1,17/03/2017,Rent,Lausanne,Rue Petit-Saint-Jean1,1003 Lausanne,"Superbe studio fra�chement r�nov� - hyper centre-ville, sans vis-�-vis. Avec salle de douche/WC, cuisini�re enti�rement agenc�e. Contact visites : B. Mezghiche - DOMICIM - barbara.mezghiche@domicim...",Appartement,1,30 m2,2.,CHF 940.�,Immostreet
2,17/03/2017,Rent,Lausanne,,1003 Lausanne,"A louer, Rue de la Mercerie 16 � Lausanne, joli studio de 19 m2, rafra�chi, situ� au 2�me �tage. Id�alement situ� au coeur de la ville de Lausanne, dans le quartier historique de la Cit�,�ce�sudio�...",Studio,1,19 m2,2.,CHF 950.�,Immostreet
3,17/03/2017,Rent,Lausanne,Rue St-Laurent 29,1003 Lausanne,"Au coeur de la ville, dans la zone pi�tonni�re. De nombreux commerces sont situ�s � proximit� imm�diate. Les bus, le m�tro ainsi que le parking de la Riponne sont � 2 minutes � pieds. Pas de buande...",Studio,1,,1.,CHF 1'010.�,Immostreet
4,17/03/2017,Rent,Lausanne,Place Chauderon 30,1003 Lausanne,"1 chambre, cuisine avec r�chaud 2 plaques, r�frig�rateur, hotte, four micro-ondes, salle de bains-wc, cave. 1 an de location : Fr. 1'130.- + 70.- charges (+3 mois de garantie bancaire = Fr. 3'390.-...",Studio,1,21 m2,5.,CHF 1'200.�,Immostreet
5,17/03/2017,Rent,Lausanne,Ruelle de Bourg 11,1003 Lausanne,"Situ� proche de toutes commodit�s et des transports publics, cet appartement de 2.5p au 2�me �tage se trouve dans un immeuble enti�rement r�nov� en 2013. Il comprend: - Cuisine �quip�e ouverte - Pi...",Appartement,2.5,44 m2,,CHF 1'990.�,Immostreet
6,17/03/2017,Rent,Lausanne,Rue St-Laurent 12,1003 Lausanne,"Charmant appartement de 3 pi�ces lumineux au centre ville de Lausanne. Comprenant un hall d'entr�e, wc s�par�, salle de bain avec baignoire et de grandes pi�ces. Cuisine agenc�e, balcon. L'immeuble...",Appartement,3,73 m2,5.,CHF 2'000.�,Immostreet
7,17/03/2017,Rent,Lausanne,Rue Charles Monnard 6,1003 Lausanne,"Bel appartement avec cuisine agenc�e (lave-vaisselle, vitroc�ram), salle de bains, WC s�par�, parquet dans les chambres, balcon , immeuble avec ascenseur. Id�alement situ� au centre de Lausanne, � ...",Appartement,3.5,99 m2,2.,CHF 2'505.�,Immostreet
8,17/03/2017,Rent,Lausanne,Rue Saint-Martin 25,1003 Lausanne,"Nous proposons � la location, un magnifique appartement de 3.5 pi�ces en duplex dans un immeuble r�cent et id�alement situ� en ville de Lausanne (proche des commerces et des transports publics). Le...",Duplex,3.5,109 m2,1.,CHF 2'630.�,Immostreet
9,17/03/2017,Rent,Lausanne,Rue Charles Monnard 6,1003 Lausanne,"Magnifique appartement de 3,5 pi�ces, 88m2 avec terrasse, vue panoraomique sur le lac, cuisine agenc�e (lave-vaisselle, vitroc�ram), grand salon avec parquet un beau mosa�que et 2 chambres. L'immeu...",Appartement,3.5,88 m2,5.,CHF 2'790.�,Immostreet
10,17/03/2017,Rent,Lausanne,Rue Centrale 15,1003 Lausanne,"Appartement de 93 m2 environ, comprenant une cuisine ouverte sur le salon, deux chambres, une salle de bain avec WC et WC s�par�. Celui-ci dispose d'une buanderie privative de 9m2 ainsi qu'une cave...",Appartement,3,93 m2,5.,CHF 2'945.�,Immostreet
11,17/03/2017,Rent,Lausanne,Rue Centrale 15,1003 Lausanne,"Centre-ville avec une grande terrasse de 30 m�. Dans un b�timent r�cemment et enti�rement r�nov� avec des mat�riaux modernes alliant �l�gance et sobri�t�, vous trouverez au dernier �tage un spacieu...",Appartement,2,110 m2,6.,CHF 2'950.�,Immostreet
12,17/03/2017,Rent,Lausanne,Rue de Gen�ve 7,1003 Lausanne,Appartement loft de 88m2 au plein coeur de Lausanne et � proximit� du m�tro M1. Ce logement offrant de grands volumes se compose comme suit: Hall meublable Wc s�par� Cuisine ouverte sur le s�jour C...,Appartement,2,90 m2,3.,CHF 3'009.�,Immostreet
13,17/03/2017,Rent,Lausanne,St-Fran�ois,1003 Lausanne,"Centre ville - St-fran�ois, immeuble r�nov� vue sur les toits. Appartements duplex meubl�s de 4,5 pi�ces� (80m2) �2chambres � coucher, salle de bain wc, salle de douche,��grand salon / salle � mang...",Appartement,4.5,120 m2,4.,CHF 3'300.�,Immostreet
14,17/03/2017,Rent,Lausanne,Beau-S�jour 8D,1003 Lausanne,"Appartement de 3.5 pi�ces de 94 m2 au 6�me �tage comprenant: Un hall avec armoires murales 2 chambres avec parquet Un s�jour Une cuisine �quip�e et agenc�e, avec lave-vaisselle et micro-ondes Une s...",Appartement,3.5,94 m2,6.,CHF 3'300.�,Immostreet
15,17/03/2017,Rent,Lausanne,Rue P�pinet 3,1003 Lausanne,"Ce beau logement se compose ainsi : Cuisine enti�rement agenc�e (frigo am�ricain, lave-vaisselle, plaques vitroc�ramiques) S�jour spacieux Grand hall avec plusieurs armoires murales Chambre spacieu...",Loft,1,165 m2,5.,CHF 3'332.�,Immostreet
16,17/03/2017,Rent,Lausanne,,1003 Lausanne,"Ce magnifique appartement de 3.5 pi�ces est situ� au coeur de la ville de Lausanne, � 20 m�tres de la place St-Fran�ois. R�nov� en 2014, ce logement est proche de toutes commodit�s (commerces, tran...",Appartement,3.5,100 m2,3.,CHF 3'500.�,Immostreet
17,17/03/2017,Rent,Lausanne,Rue Beau-S�jour 9,1003 Lausanne,"1 cuisine enti�rement agenc�e, 1 grand hall, 1 wc s�par�, 1 salle de bains-wc, 1 grand s�jour, 2 chambres, 1 terrasse, 1 cave enti�rement meubl� (vaisselle, linges, etc) 3 minutes de St-Fran�ois, 5...",Appartement,3,102 m2,1.,CHF 3'500.�,Immostreet
18,17/03/2017,Rent,Lausanne,Place Chauderon 24,1003 Lausanne,"Place Chauderon 24 - 1003 Lausanne Loft d'une pi�ce d'env. 135m2 au 6�me �tage avec tr�s belle vue Appartement lumineux avec du cachet, situ� dans un immeuble de belle architecture Quartier centre ...",Loft,1,135 m2,6.,CHF 3'500.�,Immostreet
19,17/03/2017,Rent,Lausanne,Place Chauderon 24,1003 Lausanne,"Place Chauderon 24 - 1003 Lausanne Loft d'une pi�ce d'env. 135m2 au 6�me �tage avec tr�s belle vue Appartement lumineux avec du cachet, situ� dans un immeuble de belle architecture avec ascenseur Q...",Appartement,1,135 m2,6.,CHF 3'500.�,Immostreet
20,17/03/2017,Rent,Lausanne,Passage Saint-Fran�ois 8,1003 Lausanne,"Immeuble situ� au coeur de Lausanne Appartement r�nov� en 2014 avec une vue imprenable sur la Cath�drale Ce logement b�n�ficie d'une situation id�ale au coeur de Lausanne, � proximit� des commerces...",Appartement,3.5,100 m2,3.,CHF 3'700.�,Immostreet
21,17/03/2017,Rent,Lausanne,Place de la Palud 23,1003 Lausanne,"Entr�e, d�gagement, cuisine ouverte agenc�e y.c. lave-vaisselle, salle de bains-WC, douche-WC, 3 chambres ferm�es, chemin�e de salon, acc�s direct par ascenseur.",Appartement,5,136 m2,5.,CHF 3'804.�,Immostreet
22,17/03/2017,Rent,Lausanne,Pl. de la Palud 23,1003 Lausanne,Spacieux appartement au charme fou situ� au 5�me �tage d'un immeuble datant du 15�me si�cle totalement r�nov� en 2000. Magnifique vue sur la Place de la Palud ainsi que sur la cath�drale! L'apparte...,Appartement,4,135 m2,5.,CHF 3'804.�,Immostreet
23,17/03/2017,Rent,Lausanne,Rue du Flon 8,1003 Lausanne,"MAGNIFIQUE APPARTEMENT DE 3,5 PIECES Libre de suite Immeuble situ� au Quartier du R�tillon et � proximit� imm�diate des commerces et des commodit�s. Belle superficie. Finitions soign�es et de quali...",Appartement,3.5,,2.,CHF 4'245.�,Immostreet
24,17/03/2017,Rent,Lausanne,Place Bel-Air 1,1003 Lausanne,"Cet appartement de haut standing situ� au 9�me �tage de la Tour Bel-Air, d�voilant un panorama � 360 degr�s sur la ville, le lac et les Alpes. 5 pi�ces spacieuses r�v�lent un am�nagement � la point...",Appartement,5,145 m2,9.,CHF 5'081.�,Immostreet
25,17/03/2017,Rent,Lausanne,Chemin de Mornex 7,1003 Lausanne,Chemin de Mornex 7 - 1003 Lausanne Bel appartement lumineux traversant de 7 pi�ces au 3�me �tage - env. 218 m2 1er loyer mensuel net offert. Vue panoramique sur le lac et les montagnes Quartier cen...,Appartement,7,218 m2,3.,CHF 5'600.�,Immostreet
26,17/03/2017,Rent,Lausanne,Place Bel-Air 1,1003 Lausanne,"Le Laverri�re est le joyau de la Tour Bel-Air, le plus prestigieux des appartements. Situ� � son sommet, ce duplex s'�tend sur 224 m2 habitables et jouit de finitions luxueuses. Clou du spectacle :...",Appartement,4.5,224 m2,14.,CHF 10'273.�,Immostreet
27,17/03/2017,Rent,Lausanne,Rue St-Roch 40,1004 Lausanne,"A louer au centre ville et proche de toutes commodit�s, une chambre ind�pendante avec salle de bains/WC en commun.",Appartement,1,,1.,CHF 380.�,Immostreet
28,17/03/2017,Rent,Lausanne,Av. de Beaulieu 22,1004 Lausanne,"A proximit� du centre commercial ""M�tropole2000"", ce logement comprend : - Cuisine agenc�e - Pi�ce � vivre avec du parquet - Salle de bain - Ascenseur - Cave Au pied de l'immeuble, vous pourrez pre...",Appartement,1,22 m2,4.,CHF 840.�,Immostreet
29,17/03/2017,Rent,Lausanne,Av. de Morges 76,1004 Lausanne,"Appartement r�nov� d'une pi�ce avec cuisine agenc�e, salle de bains/WC. Immeuble raccord� � la fibre optique Swisscom Proche de toutes commodit�s, commerces, transports publics, EPFL. Pour visiter ...",Appartement,1,,2.,CHF 880.�,Immostreet
30,17/03/2017,Rent,Lausanne,Av. de Morges 76,1004 Lausanne,"Appartement de 1 pi�ce au 5�me �tage comprenant une cuisine agenc�e, une salle de bain/WC, balcon. Proche de toutes commodit�s, transports publics, commerces et �coles, EPFL/UNIL. Visites: merci de...",Appartement,1,,5.,CHF 880.�,Immostreet
31,17/03/2017,Rent,Lausanne,Av. de Morges 74,1004 Lausanne,"Appartement d'une pi�ce avec cuisine agenc�e, salle de bains/WC, balcon au 2�me �tage. Proche de toutes commodit�s,commerces, transports publics, bus, EPFL. Pour visiter appeler : 079 198 61 32",Appartement,1,,2.,CHF 930.�,Immostreet
32,17/03/2017,Rent,Lausanne,Rue du Valentin 16,1004 Lausanne,"Appartement de 1 pi�ce au 2�me �tage comprenant hall d'entr�e avec armoires murales, salle de douche, pi�ce principale, cuisine agenc�e, balcon. Si l'appartement vous int�resse, merci de nous envoy...",Appartement,1,28 m2,2.,CHF 955.�,Immostreet
33,17/03/2017,Rent,Lausanne,Chemin des Rosiers 4,1004 Lausanne,"Immeuble situ� dans le quartier de Beaulieu proche des commodit�s. Studio fonctionnel avec balcon, enti�rement r�nov�. Logement compos� d'une pi�ce principale, d'une cuisine ouverte et d'une salle ...",Studio,1,16 m2,5.,CHF 960.�,Immostreet
34,17/03/2017,Rent,Lausanne,Rue de Gen�ve 75,1004 Lausanne,"Lieu id�al pour un �tudiant Partage d'appartement pour un �tudiant. Chambre (12 m2) avec salle de bain partag�e, cuisine, salon et buanderie en sous-sol. L'appartement est proche de Transports publ...",App. int�gr�,1,12 m2,1.,CHF 970.�,Immostreet
35,17/03/2017,Rent,Lausanne,Ch. Guiguer-de-Prangins 11,1004 Lausanne,Nous proposons � la location un bel appartement de 1.5 pi�ces au 4�me �tage d'un petit immeuble situ� dans un quartier calme. Le logement se compose comme suit: - Hall - Une salle de bains avec WC ...,Appartement,1.5,38 m2,4.,CHF 1'000.�,Immostreet
36,17/03/2017,Rent,Lausanne,Avenue du 24 Janvier 6,1004 Lausanne,"Immeuble proche des commodit�s, � deux pas du Palais de Beaulieu. Studio S�jour, cuisine agenc�e, salle de douches/WC, balcon et cave.",Studio,1,22 m2,R.D.C.,CHF 1'050.�,Immostreet
37,17/03/2017,Rent,Lausanne,Ch. Guiguer-de-Prangins 13,1004 Lausanne,"Nous proposons � la location, un appartement de 1.5 pi�ces dans le quartier de la Vallombreuse - Chemin de Guiguer de Prangins 13. Id�alement situ� et desservit par les transport public et le LEB, ...",Appartement,1.5,40 m2,R.D.C.,CHF 1'080.�,Immostreet
38,17/03/2017,Rent,Lausanne,Chemin des Avelines 5,1004 Lausanne,"une pi�ce parquet chevrons, cuisine agenc�e, salle de bains-WC, un balcon, une cave.",Appartement,1,25 m2,R.D.C.,CHF 1'110.�,Immostreet
39,17/03/2017,Rent,Lausanne,Ch. du Noirmont 13,1004 Lausanne,"Situ� sous le parc de Valency, ce logement se trouve dans un quartier calme avec les caract�ristiques suivantes: - Cuisine agenc�e - S�jour avec du parquet - Chambre � coucher avec du parquet - Sal...",Appartement,2,39 m2,R.D.C.,CHF 1'300.�,Immostreet
40,17/03/2017,Rent,Lausanne,Avenue de France 9,1004 Lausanne,"1 chambre, cuisine avec r�chaud 2 plaques, r�frig�rateur, hotte, four micro-ondes, salle de bains-wc, cave. 1 an de location : Fr. 1'130.- + 70.- charges (+3 mois de garantie bancaire = Fr. 3'390.-...",Studio,1,22 m2,2.,CHF 1'300.�,Immostreet
41,17/03/2017,Rent,Lausanne,AV. D'ECHALLENS 68,1004 Lausanne,"Cuisine : laboratoire / Agencement : cuisini�re �lectrique, frigo, hotte ventilation / Caract�ristiques : balcon, hall, cave . Orientation : SUD",Appartement,2.5,60 m2,3.,CHF 1'350.�,Immostreet
42,17/03/2017,Rent,Lausanne,Chemin de la Vallombreuse 5ter,1004 Lausanne,"Disponible � compter du 16 avril 2017, joli appartement de 2.5 pi�ces au 1er �tage de l'immeuble sis Chemin de la Vallombreuse 5 T � Lausanne. Au coeur d'un quartier calme, cet appartement profite ...",Appartement,2.5,,1.,CHF 1'350.�,Immostreet
43,17/03/2017,Rent,Lausanne,Rue de la Tour 5,1004 Lausanne,"Bel appartement de 1 pi�ce d'environ 47m2 situ� au 2�me �tage d'un charmant immeuble, ce dernier est compos� comme suit : - un hall d'entr�e avec de nombreux rangements - une cuisine agenc�e avec p...",Appartement,1,46 m2,2.,CHF 1'350.�,Immostreet
44,17/03/2017,Rent,Lausanne,France 64,1004 Lausanne,"Appartement compos� d'un hall d'entr�e, un s�jour, une cuisine agenc�e (sans lave-vaisselle), une chambre � coucher, une salle de bains/WC, deux balcons. Contact visites: Monsieur Grilj Jakob, loca...",Appartement,2,52 m2,4.,CHF 1'375.�,Immostreet
45,17/03/2017,Rent,Lausanne,Rue de Gen�ve 58,1004 Lausanne,"dans immeuble r�nov�, proche du centre-ville, des commerces et transports publics, cuisine agenc�e avec grand frigo, vitroc�ram, grand s�jour, salle de bains, WC s�par�, avec petite terrasse privat...",Appartement,2.5,56 m2,1.,CHF 1'390.�,Immostreet
46,17/03/2017,Rent,Lausanne,Av. de Morges 74,1004 Lausanne,"Appartement traversant de 2 pi�ces,hall , avec cuisine agenc�e ouverte, salle de bains/WC , balcon . Proche de toutes commodit�s, commerces, transports publics, UNIL/EPFL.",Appartement,2.5,,1.,CHF 1'440.�,Immostreet
47,17/03/2017,Rent,Lausanne,Chemin des Avelines 1,1004 Lausanne,"Immeuble � 2 pas des commerces et des transports publics. Appartement fonctionnel est bien situ�. Logement compos� d'une cuisine agenc�e, d'un s�jour, d'une chambre et d'une salle de bains.",Appartement,2,45 m2,4.,CHF 1'440.�,Immostreet
48,17/03/2017,Rent,Lausanne,Av. Vinet 11,1004 Lausanne,"Entr�e, d�gagement, cuisine ferm�e, frigo, salle de bains-WC, balcon.",Appartement,2.5,58 m2,R.D.C.,CHF 1'460.�,Immostreet
49,17/03/2017,Rent,Lausanne,Avenue d' Echallens 31,1004 Lausanne,"Appartement de 2.5 pi�ces au rez-de-chauss�e, comprenant : cuisine, salle de bains-WC.uisine, salle de bains-WC, ascenseur, t�l�r�seau et proche de toutes commodit�s.",Appartement,2.5,48 m2,0,CHF 1'500.�,Immostreet
50,17/03/2017,Rent,Lausanne,Ch. de Renens 43,1004 Lausanne,"Appartement de 2 pi�ces enti�rement r�nov�, comprenant hall d'entr�e avec armoire, une chambre � coucher avec d�gagement et armoire, salon, cuisine agenc�e avec grand frigo/cong�lateur, cuisini�re ...",Appartement,2,,1.,CHF 1'560.�,Immostreet
51,17/03/2017,Rent,Lausanne,Rue de Gen�ve 56,1004 Lausanne,"dans immeuble r�nov�, proche du centre-ville, des commerces et transports publics, cuisines habitables et enti�rement agenc�es avec grand frigo, vitroc�ram, salle de bains-WC, chambres � coucher av...",Appartement,3.5,73 m2,1.,CHF 1'570.�,Immostreet
52,17/03/2017,Rent,Lausanne,Avenue de Morges 68,1004 Lausanne,agr�able logement de 2 pi�ces enti�rement r�nov�.,Appartement,2,45 m2,4.,CHF 1'570.�,Immostreet
53,17/03/2017,Rent,Lausanne,Rue de Gen�ve 56,1004 Lausanne,"dans immeuble r�nov�, proche du centre-ville, des commerces et transports publics, cuisine habitable et enti�rement agenc�e avec grand frigo, vitroc�ram, salle de bains-WC, chambres � coucher avec ...",Appartement,3.5,73 m2,R.D.C.,CHF 1'600.�,Immostreet
54,17/03/2017,Rent,Lausanne,Avenue de Morges 68,1004 Lausanne,"Magnifique logement r�nov�, comprenant hall, cuisine, s�jour, chambre, salle de bains, balcon et cave",Appartement,2,45 m2,5.,CHF 1'600.�,Immostreet
55,17/03/2017,Rent,Lausanne,Avenue de Morges 68,1004 Lausanne,"magnifique logement enti�rement r�nov�, cuisine, s�jour, chambre, salle de bains et balcon.",Appartement,2,45 m2,5.,CHF 1'600.�,Immostreet
56,17/03/2017,Rent,Lausanne,Rue Pr�-du-March� 19,1004 Lausanne,"Proche du centre-ville et de toutes les commodit�s. Appartement de 2 pi�ces avec poste de conciergerie. Joli 2 pi�ces au 1er �tage avec hall d'entr�e, cuisine agenc�e et �quip�e, salle-de-bains WC,...",Appartement,2,56 m2,1.,CHF 1'610.�,Immostreet
57,17/03/2017,Rent,Lausanne,Rue du Maupas 24,1004 Lausanne,"Laissez vous s�duire par ce spacieux logement comprenant, hall, salle de bains-WC, cuisine, deux chambres, salon, deux balcons, cave et ascenceur. Parquets ponc�s et vitrifi�s et peintures rafra�ch...",Appartement,3,67 m2,6.,CHF 1'620.�,Immostreet
58,17/03/2017,Rent,Lausanne,Chemin des Rosiers 5,1004 Lausanne,Immeuble situ� dans le quartier de Beaulieu proche des commodit�s. Appartement enti�rement r�nov�. Fonctionnel avec cuisine ouverte. Logement compos� d'un s�jour carrel� et de 2 chambres dont 1 ave...,Appartement,3,55 m2,2.,CHF 1'620.�,Immostreet
59,17/03/2017,Rent,Lausanne,Avenue Davel 19,1004 Lausanne,"Hall d'entr�e, cuisine avec frigo, s�jour, 3 chambres, salle de bains, wc s�par�, balcon, cave. Location � dur�e d�termin�e jusqu'au 31 mars 2018.",Appartement,4,80 m2,R.D.C.,CHF 1'690.�,Immostreet
60,17/03/2017,Rent,Lausanne,Guiguer-de-Prangins 4,1004 Lausanne,"Joli 3 pi�ces comprenant : 1 cuisine enti�rement agenc�e (grand frigo, cuisini�re vitroc�ram), 1 salle de bain-WC, 2 chambres, 1 salon avec balcon, hall. Pas d'ascenseur. Libre de suite Si l'appart...",Appartement,3,57 m2,3.,CHF 1'730.�,Immostreet
61,17/03/2017,Rent,Lausanne,AV. JOMINI 16,1004 Lausanne,"Cuisine : fen�tre, habitable ferm�e / Agencement : frigo / Caract�ristiques : hall, terrasse, cave . Orientation : EST, NORD, OUEST",Appartement,3,61 m2,R.D.C.,CHF 1'740.�,Immostreet
62,17/03/2017,Rent,Lausanne,Avenue de France 61,1004 Lausanne,Appartement rafra�chi dans un immeuble proche du centre ville comprenant : - Cuisine d�natoire agenc�e (sans lave-vaisselle) avec balcon - S�jour avec balcon - Deux chambres avec parquet et armoire...,Appartement,3,82 m2,1.,CHF 1'770.�,Immostreet
63,17/03/2017,Rent,Lausanne,Avenue de Morges 72bis,1004 Lausanne,"Dans un quartier agr�able, proche du centre ville et de toutes commodit�s Appartement de 2.5 pi�ces au 1er �tage, comprenant : hall, cuisine, s�jour, une chambre, une salle de bains/wc et un balcon",Appartement,2.5,64 m2,1.,CHF 1'770.�,Immostreet
64,17/03/2017,Rent,Lausanne,Avenue de S�velin 13 A,1004 Lausanne,"Dans le quartier dynamique de S�beillon. Dessin�s par des architectes de renom et labellis�s �Minergie�, ce nouveau b�timent de qualit� est � proximit� du centre-ville et tr�s accessible en transpo...",Appartement,2.5,63 m2,R.D.C.,CHF 1'770.�,Immostreet
65,17/03/2017,Rent,Lausanne,Avenue de France 69,1004 Lausanne,"Cosy appartement de 2 pi�ces, dans les combles d'une ancienne maison, avec jolies poutres, comprenant : entr�e, une spacieuse chambre � coucher, lumineux s�jour, cuisine agenc�e ouverte et salle de...",Attique,2,65 m2,5.,CHF 1'779.�,Immostreet
66,17/03/2017,Rent,Lausanne,Avenue de S�velin 13 B,1004 Lausanne,"Dans le quartier dynamique de S�beillon. Dessin�s par des architectes de renom et labellis�s �Minergie�, ce nouveau b�timent de qualit� est � proximit� du centre-ville et tr�s accessible en transpo...",Appartement,2.5,63 m2,R.D.C.,CHF 1'790.�,Immostreet
67,17/03/2017,Rent,Lausanne,Ch. de Renens 41,1004 Lausanne,"Magnifique appartement traversant avec vue sur le lac enti�rement r�nov� comprenant 2 pi�ces, hall meublable avec armoire , grande cuisine agenc�e avec grand frigo/cong�lateur , lave vaisselle, hot...",Appartement,2.5,60 m2,4.,CHF 1'800.�,Immostreet
68,17/03/2017,Rent,Lausanne,Av. de Morges 76,1004 Lausanne,"Joli appartement de 3 pi�ces comprenant entr�e avec armoires , grand salon, 2 chambres � coucher, cuisine agenc�e, salle de bains/WC, grand balcon. Proche de toutes commodit�s,UNIL,EPFL. Pour visit...",Appartement,3,,4.,CHF 1'825.�,Immostreet
69,17/03/2017,Rent,Lausanne,Rue de Gen�ve 56,1004 Lausanne,"dans immeuble r�nov�, proche du centre-ville, des commerces et transports publics, cuisines habitables et enti�rement agenc�es avec grand frigo, vitroc�ram, salle de bains-WC, chambres � coucher av...",Appartement,4.5,89 m2,1.,CHF 1'905.�,Immostreet
70,17/03/2017,Rent,Lausanne,Rue de Gen�ve 56,1004 Lausanne,"dans immeuble r�nov�, proche du centre-ville, des commerces et transports publics, cuisine habitable et enti�rement agenc�e avec grand frigo, vitroc�ram, salle de bains-WC, chambres � coucher avec ...",Appartement,4.5,89 m2,R.D.C.,CHF 1'945.�,Immostreet
71,17/03/2017,Rent,Lausanne,Av. de Morges 25A,1004 Lausanne,"Lausanne centre ville, pied-�-terre meubl�, agenc� pour une personne, sans animaux. Situ� dans un parc priv� de 4000m2, pr�s de toutes commodit�s. Les personnes int�ress�es sont pri�es de contacter...",Maison individuelle,,,,CHF 2'000.�,Immostreet
72,17/03/2017,Rent,Lausanne,Av. de France 98,1004 Lausanne,"Dans nouvel immeuble, joli appartement de 2,5 pi�ces au rez, 68 m2, compos� d'une entr�e, d'une cuisine agenc�e, d'un grand s�jour, d'une salle de douche / WC �quip�e d'un lave-linge et d'un s�che-...",Appartement,2.5,68 m2,R.D.C.,CHF 2'050.�,Immostreet
73,17/03/2017,Rent,Lausanne,Av. de S�very 12,1004 Lausanne,"A proximit� de toutes commodit�s et des transports publics, joli 3,5 pi�ces au 3�me �tage d'un petit immeuble, comprenant : - Hall - 2 chambres - Salon - Cuisine agenc�e - Salle de bains/WC - Balco...",Appartement,3.5,73 m2,3.,CHF 2'080.�,Immostreet
74,17/03/2017,Rent,Lausanne,Ch. de Pierrefleur 35,1004 Lausanne,"Spacieux et lumineux appartement de 4.5 pi�ces au 1er �tage dans immeuble locatif, se composant comme suit : - Hall avec armoires murales - Grand s�jour lumineux avec acc�s direct au balcon - Cuisi...",Appartement,4.5,101 m2,1.,CHF 2'100.�,Immostreet
75,17/03/2017,Rent,Lausanne,Rue Couchirard 9,1004 Lausanne,"Dans un quartier proche des transports publics et des commerces. A proximit� des commodit�s. Spacieux logement compos� de deux chambres, d'un hall, d'un s�jour, d'une cuisine agenc�e, d'une salle d...",Appartement,3.5,84 m2,2.,CHF 2'120.�,Immostreet
76,17/03/2017,Rent,Lausanne,Avenue de France 18A,1004 Lausanne,"A louer pour le 16 avril 2017, magnifique appartement de 3.5 pi�ces au 3�me �tage, proche des transports publics et des commodit�s. Ce dernier comprend : - Hall - Cuisine agenc�e - S�jour avec parq...",Appartement,3.5,74 m2,3.,CHF 2'145.�,Immostreet
77,17/03/2017,Rent,Lausanne,Chemin du Risoux 1,1004 Lausanne,"Quartier de Mont�tan, bus no 7 et centre commercial COOP � 5 minutes � pied. Tr�s calme, en bordure parc public. Immeuble enti�rement r�nov� avec isolation des fa�ades. Comprenant hall, s�jour, 2 c...",Appartement,3,80 m2,0,CHF 2'150.�,Immostreet
78,17/03/2017,Rent,Lausanne,"Boisy, chemin du43",1004 Lausanne,"Dans un immeuble de bon standing, ce bel appartement lumineux de 3 � pi�ces au rez-de-chauss�e sup�rieur, rafra�chissement des peintures effectu�, se compose comme suit: Salon, cuisine semi-ouverte...",Appartement,3.5,78 m2,R.D.C.,CHF 2'190.�,Immostreet
79,17/03/2017,Rent,Lausanne,Avenue de S�velin 13 C,1004 Lausanne,"Dans le quartier dynamique de S�beillon. Dessin�s par des architectes de renom et labellis�s �Minergie�, ce nouveau b�timent (2016) de qualit� est � proximit� du centre-ville et tr�s accessible en ...",Appartement,3.5,77 m2,4.,CHF 2'225.�,Immostreet
80,17/03/2017,Rent,Lausanne,Ch de Renens,1004 Lausanne,"Cet appartement avec beaucoup de cachet a �t� enti�rement r�nov� . Il est situ� dans un petit immeuble pr�s du centre ville. Il dispose d�une seule chambre � coucher ferm�e avec dressing, d�une mez...",Appartement,3.5,76 m2,2.,CHF 2'290.�,Immostreet
81,17/03/2017,Rent,Lausanne,"95, rue de Gen�ve",1004 Lausanne,"S�beillon 356 logements neufs dans un quartier convivial � la route de Gen�ve � Lausanne, au c?ur du r�seau urbain. Derri�re d'�l�gantes fa�ades en pierre des Grisons, ce nouvel ensemble de cinq im...",Appartement,4.5,101 m2,1.,CHF 2'310.�,Immostreet
82,17/03/2017,Rent,Lausanne,Ch. des Clochetons 37,1004 Lausanne,"Spacieux appartement de 3,5 pi�ces de 96 m2 compos� d'une entr�e, de deux chambres � coucher, d'une salle de bains / WC, d'une salle de douche / WC, d'une cuisine agenc�e ouverte sur le salon / sal...",Appartement,3.5,96 m2,,CHF 2'380.�,Immostreet
83,17/03/2017,Rent,Lausanne,Vinet 16,1004 Lausanne,"Appartement de 2.5 pi�ces au 3�me �tage compos� d'un hall d'entr�e, un s�jour, une cuisine agenc�e, une chambre � coucher, une salle de bains/WC, un WC s�par�, un balcon. CHF 2'400.00 charges compr...",Appartement,2.5,77 m2,3.,CHF 2'400.�,Immostreet
84,17/03/2017,Rent,Lausanne,"95, rue de Gen�ve",1004 Lausanne,"S�beillon 356 logements neufs dans un quartier convivial � la route de Gen�ve � Lausanne, au c?ur du r�seau urbain. Derri�re d'�l�gantes fa�ades en pierre des Grisons, ce nouvel ensemble de cinq im...",Appartement,4.5,101 m2,4.,CHF 2'420.�,Immostreet
85,17/03/2017,Rent,Lausanne,Chemine des Aub�pines 15,1004 Lausanne,"Lausanne centre � 5 min � pieds, de Chauderon, mais sans nuisances. Vous cherchez...une situation unique ? Splendide attique en sur-combles. Vue panoramique sur le lac et les Alpes. Superbe terrass...",Appartement,2.5,70 m2,4.,CHF 2'500.�,Immostreet
86,17/03/2017,Rent,Lausanne,Av. d'Echallens 8,1004 Lausanne,"Immeuble de 1900, proche de toutes commodit�s. Ce grand appartement avec cachet est tr�s lumineux et se compose comme suit: - hall donnant sur salle � manger - 2 chambres � coucher - un salon/salle...",Appartement,4.5,100 m2,2.,CHF 2'550.�,Immostreet
87,17/03/2017,Rent,Lausanne,Rue de Gen�ve 77bis,1004 Lausanne,"La rue de Gen�ve 77 bis est une rue anim�e de Lausanne, situ� non loin du centre-ville et � proximit� des transports publics. De nombreux commerces sont situ�s � proximit� imm�diate. Ce magnifique ...",Appartement,4.5,110 m2,,CHF 2'590.�,Immostreet
88,17/03/2017,Rent,Lausanne,Avenue de S�velin 13 A,1004 Lausanne,"Dans le quartier dynamique de S�beillon. Dessin�s par des architectes de renom et labellis�s �Minergie�, ce nouveau b�timent de qualit� est � proximit� du centre-ville et tr�s accessible en transpo...",Appartement,4.5,99 m2,1.,CHF 2'630.�,Immostreet
89,17/03/2017,Rent,Lausanne,Rue de l'Ale 6,1004 Lausanne,"En plein coeur de Lausanne, dans une rue commer�ante, se bel attique en duplex ce compose de la fa�on suivante: Au niveau inf�rieur, un hall d'entr�e donne acc�s a deux chambres dont une avec un gr...",App. mansard�,3,122 m2,,CHF 2'733.�,Immostreet
90,17/03/2017,Rent,Lausanne,Av. de Morges 31,1004 Lausanne,"Logement au 1er �tage, comprenant s�jour avec chemin�e, grande cuisine non agenc�e, grand hall, 4 chambres dont certaines avec armoire murale, sallle-de-bains, WC s�par�, cave, balcon. � louer pour...",Appartement,5,107 m2,1.,CHF 2'800.�,Immostreet
91,17/03/2017,Rent,Lausanne,Rue du Valentin 34,1004 Lausanne,"Entr�e, d�gagement, cuisine ouverte enti�rement agenc�e y.c. lave-vaisselle, coin � manger, salle de bains-WC, WC s�par�, balcon",Appartement,4.5,95 m2,11.,CHF 2'880.�,Immostreet
92,17/03/2017,Rent,Lausanne,Rue de la Tour 33,1004 Lausanne,"Immeuble situ� au centre-ville, entre deux rues pi�tonnes et marchandes, proche de toutes commodit�s. Appartement en duplex comprenant: hall d'entr�e, s�jour avec chemin�e, 3 chambres � coucher, cu...",Appartement,4.5,118 m2,2.,CHF 2'910.�,Immostreet
93,17/03/2017,Rent,Lausanne,Chemin des Retraites 11,1004 Lausanne,Appartement de 3.5 pi�ces au 6�me �tage comprenant: - un hall - un wc s�par� - une cuisine enti�rement agenc�e avec lave-linge combin� s�che-linge et ouverte sur un tr�s grand s�jour - une chambre ...,Attique,3.5,90 m2,6.,CHF 2'950.�,Immostreet
94,17/03/2017,Rent,Lausanne,Avenue de France 1,1004 Lausanne,"Avenue de France 1 � Lausanne Loft au 5�me �tage - env. 116 m2 Quartier centre ville, proche des transports publics et commerces �Cuisine agenc�e et �quip�e ouverte �Mezzanine �Salle de bains/wc av...",Loft,1,116 m2,5.,CHF 3'080.�,Immostreet
95,17/03/2017,Rent,Lausanne,Avenue de France,1004 Lausanne,"Situ� au coeur de la ville de Lausanne, � deux pas de la place Chauderon, proche des commerces, des restaurants et transports publics, ce magnifique appartement de 4.5 pi�ces se situe au 4�me �tage...",Appartement,4.5,109 m2,4.,CHF 3'150.�,Immostreet
96,17/03/2017,Rent,Lausanne,Chemin Aim�-Steinlen 15,1004 Lausanne,L'appartement est spacieux et lumineux et se compose comme suit: - hall d'entr�e avec rangements - cuisine ouvert enti�rement �quip�e (vitroc�ramique) ouvert sur grand s�jour/salle � manger - 3 cha...,Appartement,4.5,110 m2,4.,CHF 3'220.�,Immostreet
97,17/03/2017,Rent,Lausanne,,1004 Lausanne,"Appartement de 4.5 pi�ces (117m2) dans immeuble neuf. Equip� de tout le confort, notamment cuisine, salles de bains, fibre optique (t�l�phonie et web). Construit en d�veloppement durable et minergi...",Appartement,4.5,117 m2,4.,CHF 3'300.�,Immostreet
98,17/03/2017,Rent,Lausanne,Aim�-Steinlen 15,1004 Lausanne,"Appartement de 4.5 pi�ces au 4�me �tage compos� d'un hall d'entr�e, d'un s�jour, d'une cuisine agenc�e, de trois chambres � coucher, d'une salle de bains/WC, d'une salle de douche/WC, d'un balcon. ...",Appartement,4.5,107 m2,4.,CHF 3'470.�,Immostreet
99,17/03/2017,Rent,Lausanne,"Jomini, avenue5",1004 Lausanne,"Situ� proche de toutes les commodit�s, �coles et axes autoroutiers � 5 min, ce bel appartement se compose comme suit: Hall d'entr�e Salon Salle � manger partiellement ouverte sur le salon Cuisine o...",Appartement,4.5,100 m2,2.,CHF 3'700.�,Immostreet
100,17/03/2017,Rent,Lausanne,Avenue d'Echallens 22,1004 Lausanne,"Hall, cuisine agenc�e, y.c. lave-vaisselle, salle de bains/WC att. � une des chambres, salle de bains/WC.",Appartement,4.5,144 m2,6.,CHF 4'000.�,Immostreet
101,17/03/2017,Rent,Lausanne,Rue du Petit-Valentin 6,1004 Lausanne,Immeuble � 5 minutes de la Riponne cave au sous-sol,Cave,,,,CHF 50.�,Immostreet
102,17/03/2017,Rent,Lausanne,Rue Pr�-du-March� 19,1004 Lausanne,Proche du centre-ville et de toutes les commodit�s.,Cave,,,,CHF 60.�,Immostreet
103,17/03/2017,Rent,Lausanne,Rue du Tunnel 9,1005 Lausanne,"Immeuble id�alement situ� au centre-ville, proche des transports publics et de toutes commodit�s. Appartement 1 pi�ce au 4�me �tage compos� d'une chambre, une cuisine, une salle-de-bains/wc, et une...",Appartement,1,28 m2,,CHF 800.�,Immostreet
104,17/03/2017,Rent,Lausanne,Rue St-Martin 28,1005 Lausanne,"Vous recherchez un logement proche des commerces, des transports publics. Cet appartement est fait pour vous!!! Pi�ce � vivre avec parquet au sol Kitchenette r�nov�e avec frigo, four et plan de cui...",Appartement,1,21 m2,2.,CHF 840.�,Immostreet
105,17/03/2017,Rent,Lausanne,Rue du Tunnel 11,1005 Lausanne,"Immeuble id�alement situ�, � proximit� des transports publics (arr�t M2), commerces et �coles. Appartement 1 pi�ce au 2�me �tage compos� d'une chambre, une cuisinette, une salle-de-bains/wc, une ca...",Appartement,1,18 m2,2.,CHF 970.�,Immostreet
106,17/03/2017,Rent,Lausanne,Avenue des Mousquines 23,1005 Lausanne,"Petit studio meubl� dans le centre le Lausanne, meubl�. Il a aussi une cave. 3�me �tage avec ascenseur.",Studio,1,18 m2,3.,CHF 990.�,Immostreet
107,17/03/2017,Rent,Lausanne,Avenue de la D�le 9,1005 Lausanne,"situ� dans immeuble � proximit� du CHUV et de la station ""place de l'Ours"" du m�tro M2, cuisine agenc�e, salle de bains-WC, balcon.",Appartement,1,26 m2,5.,CHF 1'060.�,Immostreet
108,17/03/2017,Rent,Lausanne,Chemin du Calvaire 17,1005 Lausanne,Renseignement chez Mme Goldschmidt t�l: 021 340 66 66 le matin de 8.30 h � 12.30 h,Appartement,1,26 m2,R.D.C.,CHF 1'070.�,Immostreet
109,17/03/2017,Rent,Lausanne,Rue Saint-Martin 36,1005 Lausanne,"Centre ville, proche des commerces et des transports publics, hall, cuisine agenc�e, salle de bains-WC.",Appartement,2,48 m2,2.,CHF 1'240.�,Immostreet
110,17/03/2017,Rent,Lausanne,Pace du Vallon,1005 Lausanne,"Joli appartement au centre de Lausanne � l'abri de la pollution sonore, situ� au rez sup�rieur comprend : * 1er hall d'entr�e face � une cuisine moderne agenc�e,���� agr�able et ferm�e ; * 2� hall ...",Appartement,2.5,,R.D.C.,CHF 1'360.�,Immostreet
111,17/03/2017,Rent,Lausanne,Chemin de Montmeillan 19 -21,1005 Lausanne,"Immeuble partiellement r�nov�, avec ascenseur, secteur calme et verdoyant, proche du centre et des commodit�s. Entr�e, s�jour avec rangements, une grande chambre, cuisine habitable, r�cente s�par�e...",Appartement,2,43 m2,4.,CHF 1'415.�,Immostreet
112,17/03/2017,Rent,Lausanne,Rue du Tunnel 18,1005 Lausanne,"Entr�e, cuisine agenc�e, salle de bains-WC.",Appartement,2,45 m2,0,CHF 1'550.�,Immostreet
113,17/03/2017,Rent,Lausanne,C�sar-Roux 4,1005 Lausanne,"Appartement de 2.5 pi�ces au 6�me �tage compos� d'un hall d'entre, un s�jour, une cuisine agenc�e, une chambre � coucher, une salle de bains/WC et un balcon. Pour les visites, merci de contacter la...",Appartement,2.5,66 m2,6.,CHF 1'620.�,Immostreet
114,17/03/2017,Rent,Lausanne,Avenue Eug�ne-Rambert 5,1005 Lausanne,Un cadre de vie agr�able vous attend dans ce bel appartement disposant d'une situation de choix dans un quartier joliment arboris�. Les transports publics � proximit� vous permettront d'acc�der en ...,Appartement,3.5,66 m2,3.,CHF 1'700.�,Immostreet
115,17/03/2017,Rent,Lausanne,Av. de la d�le 10,1005 Lausanne,"Appartement de 3.5 pi�ces au 4e �tage d'env. 65 m2, comprenant: hall d'entr�e, cuisine agenc�e, 2 chambres, s�jour, salle-de-bains/WC, balcon",Appartement,3.5,65 m2,4.,CHF 1'700.�,Immostreet
116,17/03/2017,Rent,Lausanne,Avenue de B�thusy 28,1005 Lausanne,Pour les visites nous vous prions de vous inscrire aupr�s de Madame Popa Potalivo au 078/708.42.93. Les visites group�es auront lieu: le lundi 20 mars et le jeudi 23 mars de 17h00 � 19h00. Ce beau ...,Duplex,2.5,62 m2,5.,CHF 1'725.�,Immostreet
117,17/03/2017,Rent,Lausanne,C�sar-Roux 20,1005 Lausanne,Lausanne C�sar-Roux 20 ? Appartement 3 pi�ces au rez inf�rieur Libre de suite Loyer Fr. 1490.-- plus Fr. 240.-- charges Pour visiter : Immeubles Services Plus au 021/964.46.36,Appartement,3,,R.D.C.,CHF 1'730.�,Immostreet
118,17/03/2017,Rent,Lausanne,C�sar-Roux 4,1005 Lausanne,"Appartement de 2.5 pi�ces au 1er �tage compos� comme suit: Un salon, une chambre, une cuisine agenc�e, une salle de bains/WC, un r�duit et un balcon. Pour les visites, deux visites group�es sont pr...",Appartement,2.5,66 m2,1.,CHF 1'740.�,Immostreet
119,17/03/2017,Rent,Lausanne,Orient-Ville8,1005 Lausanne,"Bel appartement de 2� pi�ces au 1er �tage r�nov� dans immeuble ancien, beaucoup de cachet, � deux pas du Parc Mon-Repos. Tr�s tranquille. - Hall, s�jour, cuisine agenc�e, salle de bains/WC, chambre...",Appartement,2.5,66 m2,1.,CHF 1'780.�,Immostreet
120,17/03/2017,Rent,Lausanne,Av. du L�man 42,1005 Lausanne,"Entr�e, r�duit, cuisine ferm�e, frigo, cuisini�re �lectrique, hotte, salle de bains-WC, balcon.",Appartement,3,70 m2,R.D.C.,CHF 1'870.�,Immostreet
121,17/03/2017,Rent,Lausanne,Avenue du L�man 68,1005 Lausanne,"Lausanne, appartement de 3.5 pi�ces au 4�me �tage comprenant une entr�e, une cuisine agenc�e, un s�jour, une salle de bains/WC, deux chambres � coucher et un balcon. Une place de parc disponible � ...",Appartement,,76 m2,,CHF 1'965.�,Immostreet
122,17/03/2017,Rent,Lausanne,Avenue du L�man 28,1005 Lausanne,"Immeuble id�alement situ� � 5 minutes du centre ville. Bel appartement avec beaucoup de rangements orient� sud, offrant une vue sur le lac. Comprenant: - Une cuisine - Un grand s�jour ouvert sur le...",Appartement,3.5,68 m2,3.,CHF 2'030.�,Immostreet
123,17/03/2017,Rent,Lausanne,Avenue du L�man 28,1005 Lausanne,Immeuble id�alement situ� � 5 minutes du centre ville. Bel appartement id�al pour jeune couple ou famille Logement spacieux et lumineux comprenant: - Une cuisine agenc�e - Un grand s�jour avec un c...,Appartement,3.5,68 m2,5.,CHF 2'060.�,Immostreet
124,17/03/2017,Rent,Lausanne,Avenue de B�thusy 4,1005 Lausanne,"Dans un immeuble proche du centre-ville, proche de toutes commodift�s (acc�s au M2 � deux pas). Appartement de 3.5 pi�ces Magnifique logement comprenant: - un hall - un s�jour - 2 chambres � couche...",Appartement,3.5,73 m2,2.,CHF 2'215.�,Immostreet
125,17/03/2017,Rent,Lausanne,Rue St-Martin 28,1005 Lausanne,"Vous recherchez un logement proche des commerces, des transports publics Cet appartement lumineux est fait pour vous... - S�jour avec parquet au sol r�nov� -Cuisine enti�rement r�nov�e - Pour votre...",Appartement,4.5,78 m2,3.,CHF 2'245.�,Immostreet
126,17/03/2017,Rent,Lausanne,Rue St-Martin 30,1005 Lausanne,"Vous recherchez un logement proche des commerces, des transports publics Cet appartement lumineux est fait pour vous... S�jour avec parquet au sol r�nov� Cuisine enti�rement r�nov�e Pour votre conf...",Appartement,4.5,77 m2,3.,CHF 2'245.�,Immostreet
127,17/03/2017,Rent,Lausanne,Charles-Vuillermet 6,1005 Lausanne,Bel immeuble en face de la cath�drale au coeur du quartier de la Cit� Charmant appartement en duplex id�al pour jeune couple Comprenant: - Hall d'entr�e avec armoires murales - Cuisine agenc�e ouve...,Appartement,3,90 m2,4.,CHF 2'290.�,Immostreet
128,17/03/2017,Rent,Lausanne,Av. de Jaman 13,1005 Lausanne,L'appartement se trouve au rez-de-chauss�e d'un immeuble r�sidentiel situ� en plein coeur de Lausanne dans un quartier calme et r�sidentiel!. Il se compose de 3.5 pi�ces avec une surface habitable ...,Appartement,3.5,83 m2,,CHF 2'340.�,Immostreet
129,17/03/2017,Rent,Lausanne,Av. de Rumine 22,1005 Lausanne,"A louer beau et spacieux 4 pi�ces, d�environ 125 m2, comprenant : - hall d'entr�e - cuisine agenc�e habitable - salle � manger - spacieux salon avec chemin�e d'ornement - salle de bains avec lavabo...",Appartement,4,127 m2,5.,CHF 2'610.�,Immostreet
130,17/03/2017,Rent,Lausanne,Acad�mie 2,1005 Lausanne,"Spacieux logement en duplex avec cachet, situ� au c�ur de la cit� de Lausanne, � proximit� de toutes commodit�s. Descriptif : Grand s�jour avec chemin�e 2 chambres Cuisine agenc�e et habitable Sall...",Appartement,3.5,85 m2,2.,CHF 2'640.�,Immostreet
131,17/03/2017,Rent,Lausanne,Montagibert 2,1005 Lausanne,"Appartement de 4 pi�ces au 4�me �tage compos� d'un hall d'entr�e, un s�jour, une cuisine agenc�e, trois chambres � coucher, une salle de bain/WC, deux balcons. Pour les visites, merci de contacter ...",Appartement,4,87 m2,4.,CHF 2'885.�,Immostreet
132,17/03/2017,Rent,Lausanne,Av. de Montoie 37,1005 Lausanne,"Au 5e �tage, spacieux appartement de 4,5 pi�ces compos� d'une grande entr�e avec armoire murale, d'une cuisine agenc�e ferm�e, d'un salon / salle � manger, de trois chambres, d'une salle de bains /...",Appartement,4.5,116 m2,,CHF 2'900.�,Immostreet
133,17/03/2017,Rent,Lausanne,Avenue du L�man89,1005 Lausanne,"Appartement compos� de 3.5 pi�ces au 1er �tage d'env. 92m2 avec balcon, hall d'entr�e, cuisine agenc�e ouverte sur salon/salle � manger, une salle de douche, ainsi que deux chambres dont une avec s...",Appartement,3.5,92 m2,1.,CHF 2'900.�,Immostreet
134,17/03/2017,Rent,Lausanne,Avenue du L�man89,1005 Lausanne,"Appartement de 3.5 pi�ces au 3�me �tage compos� d'un hall d'entr�e, un s�jour, une cuisine agenc�e, deux chambres � coucher, une salle de bains/WC, une salle de douche/WC, un balcon. Charges et pla...",Appartement,3.5,92 m2,3.,CHF 2'900.�,Immostreet
135,17/03/2017,Rent,Lausanne,"Rumine 21, av. de",1005 Lausanne,Cet appartement luminueux et agr�ablement am�nag� saura vous s�duire par : - Large hall avec penderie et cagibi - Cuisine �quip�e avec porte coulissante sur la salle � manger - S�jour et salle � ma...,Appartement,5,111 m2,4.,CHF 2'945.�,Immostreet
136,17/03/2017,Rent,Lausanne,Avenue du Tribunal-F�d�ral 23,1005 Lausanne,"Ce prestigieux appartement, enti�rement r�nov�, se situe au 2�me �tage d'un immeuble du quartier de la Place de l'Ours, � c�t� de la piscine de Mon-Repos, du parc, du centre ville et des transports...",Appartement,4,108 m2,2.,CHF 2'950.�,Immostreet
137,17/03/2017,Rent,Lausanne,AV. EUGENE-RAMBERT 24,1005 Lausanne,"Cuisine : fen�tre, habitable ferm�e / Agencement : cong�lateur, cuisini�re �lectrique, frigo, hotte ventilation, lave-vaisselle / Caract�ristiques : balcon, hall, chemin�e ornement, cave . Orientat...",Appartement,5.5,129 m2,2.,CHF 3'020.�,Immostreet
138,17/03/2017,Rent,Lausanne,Avenue de Rumine,1005 Lausanne,"Centre ville - downtown Appartement dans les combles, vue sur le lac, grand s�jour, hall, 2 chambres � coucher, salle de bain, cuisine, cave, jardin commun � l'immeuble � disposition 90m2 (env.). P...",Appartement,3.5,90 m2,3.,CHF 3'250.�,Immostreet
139,17/03/2017,Rent,Lausanne,Chemin de la Vuach�re 2 A,1005 Lausanne,"dans immeuble r�cent et r�sidentiel de quatre logements, cuisine enti�rement agenc�e, salle de bains-WC, douche-WC, balcon avec vue imprenable sur le lac.",Appartement,5,150 m2,1.,CHF 3'780.�,Immostreet
140,17/03/2017,Rent,Lausanne,Avenue du L�man 34,1005 Lausanne,"Appartement comprenant : grand s�jour avec cuisine enti�rement agenc�e ouverte et neuve, 4 chambres � coucher, une salle de bains avec douche, une salle de bains avec baignoire et un WC s�par�. Deu...",Appartement,5,170 m2,6.,CHF 4'450.�,Immostreet
141,17/03/2017,Rent,Lausanne,Avenue du L�man 30,1005 Lausanne,"Magnifique appartement traversant avec balcons comprenant : un grand s�jour de 40 m2, 3 chambres � coucher, cuisine agenc�e et �quip�e, 2 salles de bains + 1 WC s�par�. Une cave. Libre au 1er avril...",Appartement,5,162 m2,7.,CHF 4'510.�,Immostreet
142,17/03/2017,Rent,Lausanne,Av. de Rumine 4,1005 Lausanne,"A louer dans immeuble de standing appartement de 212 m2 au 1er �tage comprenant : hall, huit pi�ces, une chemin�e de salon, cuisine agenc�e avec cuisini�re vitroceram avec four et four � vapeur, fr...",Appartement,8.5,212 m2,1.,CHF 4'775.�,Immostreet
143,17/03/2017,Rent,Lausanne,Avenue Secr�tan 41,1005 Lausanne,"situ� dans un quartier calme et verdoyant, proche des commerces et des transports, 1 grand hall, cuisine enti�rement agenc�e, salle de bains-WC, douche-WC, chemin�e, balcon, jouissance partielle du...",Appartement,7.5,200 m2,1.,CHF 4'920.�,Immostreet
144,17/03/2017,Rent,Lausanne,Avenue des Mousquines 18,1005 Lausanne,"Avenue des Mousquines 18 � Lausanne. Magnifique appartement de 6.5 pi�ces, enti�rement r�nov� avec des mat�riaux de qualit�. Grands et beaux volumes avec luminosit� optimale, hauts plafonds, magnif...",Appartement,6.5,151 m2,1.,CHF 5'650.�,Immostreet
145,17/03/2017,Rent,Lausanne,Avenue Verdeil 16,1005 Lausanne,"Dans les beaux quartiers de Lausanne, cette charmante villa � l'abri des regards et de toute nuisance avec une somptueuse vue sur le lac, profite d'un cadre de vie agr�able. Cette propri�t� a fait ...",Villa,5.5,257 m2,,CHF 7'100.�,Immostreet
146,17/03/2017,Rent,Lausanne,Avenue Verdeil 16,1005 Lausanne,"Maginifique villa de 5.5 pi�ces avec vue sur le lac Tr�s belle villa de 5.5 pi�ces, quartier Verdeil � Lausanne Cette tr�s belle villa mitoyenne d'environ 257m2 b�n�ficie d'un emplacement privil�gi...",Appartement,5.5,257 m2,R.D.C.,CHF 7'100.�,Immostreet
147,17/03/2017,Rent,Lausanne,Passage Perdonnet 1,1005 Lausanne,"Pour de suite ou � convenir, nous louons : - D�p�t � CHF 60.00 + 8% TVA Nous nous r�jouissons de votre prise de contact!",Pi�ce pour les hobbys,,,,CHF 60.�,Immostreet
148,17/03/2017,Rent,Lausanne,Rue St-Martin 32,1005 Lausanne,Spacieux local archives,Pi�ce pour les hobbys,,,,CHF 605.�,Immostreet
149,17/03/2017,Rent,Lausanne,Montolivet 19,1006 Lausanne,Annonce adress�e uniquement � la gente f�minine. Disponibilit� : 15 FEVRIER 2017 VISITE : contacter M. Milloud le concierge au 079.213.93.18 Chambres ind�pendantes meubl�es comprenant : - Frigo - L...,Chambre,1,,3.,CHF 590.�,Immostreet
150,17/03/2017,Rent,Lausanne,Chemin de Chissiez 7,1006 Lausanne,"Immeuble proche de toutes commodit�s. Ce logement se compose d'un s�jour lumineux, d'une cuisine r�cente s�par�e enti�rement �quip�e et d'une salle de douche.",Studio,1,,1.,CHF 750.�,Immostreet
151,17/03/2017,Rent,Lausanne,Avenue des Alpes 22,1006 Lausanne,"Comprenant: grand hall, s�jour avec chemin�e, salle � manger, cuisine ferm�e habitable, r�duit, 5 chambres � coucher avec armoires murales, 2 salles d'eau dont une avec colonne de lavage. Une cave....",Appartement,1,,1.,CHF 880.�,Immostreet
152,17/03/2017,Rent,Lausanne,Chemin de Chandieu 22,1006 Lausanne,Appartement de 1.5 pi�ces au 2�me �tage situ� dans le quartier privil�gi� de Montchoisi entre la Gare et le Lac 32 m2 Hall entr�e Cuisine ferm�e et �quip�e 1 chambre/salon Balon Salle de bains /wc ...,Appartement,1.5,,2.,CHF 960.�,Immostreet
153,17/03/2017,Rent,Lausanne,Bd de Grancy 35,1006 Lausanne,"Joli studio fonctionnel avec un hall, cuisine semie-agenc�e, salle-de-douche et pi�ce principale. Attention ! contrat �tabli pour une dur�e fixe ne d�passant pas le 31.03.2019.",Appartement,1,26 m2,5.,CHF 995.�,Immostreet
154,17/03/2017,Rent,Lausanne,Chemin de Vermont 20,1006 Lausanne,"Cuisine agenc�e, salle-de-bains-wc, s�jour, chambre, balcon.",Appartement,2.5,55 m2,R.D.C.,CHF 1'300.�,Immostreet
155,17/03/2017,Rent,Lausanne,Avenue des Alpes 10,1006 Lausanne,"Ce loft neuf tr�s lumineux se situe au rez-inf�rieur d'un petit immeuble tranquille du d�but 1900. L�immeuble, r�nov� en 2015, est id�alement situ� � 10 minutes � pied de la gare, � un jet de pierr...",Appartement,,58 m2,R.D.C.,CHF 1'615.�,Immostreet
156,17/03/2017,Rent,Lausanne,Chemin du Cap 1,1006 Lausanne,"hall d'entr�e, s�jour, 1 chambre, cuisine agenc�e (cuisini�re, frigo, four, hotte de ventilation), salle de bains, WC s�par�, balcon, cave.",Appartement,2,,2.,CHF 1'650.�,Immostreet
157,17/03/2017,Rent,Lausanne,Av. Montchoisi 47,1006 Lausanne,"Comprenant: hall, deux chambres, salon, cuisine agenc�e, WC s�par�, salle-de-bains.",Appartement,3.5,80 m2,3.,CHF 1'670.�,Immostreet
158,17/03/2017,Rent,Lausanne,Rue du Cr�t 10,1006 Lausanne,"Logement traversant dans un charmant petit immeuble en face de la place de Milan comprenant : - Cuisine agenc�e, ferm�e - S�jour avec parquet et acc�s � une terrasse privative - Deux chambres avec ...",Appartement,3,69 m2,R.D.C.,CHF 1'680.�,Immostreet
159,17/03/2017,Rent,Lausanne,avenue d'ouchy,1006 Lausanne,"A louer charmant et lumineux appartement situ� � l'avenue d'Ouchy � Lausanne. Situ� � deux pas du lac et � proximit� du m2, des commerces et restaurants. Hall d'entr�e, salon-s�jour, cuisine, une c...",Appartement,2.5,60 m2,3.,CHF 1'750.�,Immostreet
160,17/03/2017,Rent,Lausanne,CH. DU TRABANDAN 37 C,1006 Lausanne,"Cuisine : laboratoire, ouverte s/coin manger / Agencement : cuisini�re �lectrique, frigo, hotte ventilation, lave-vaisselle / Caract�ristiques : balcon, hall, cave . Orientation : OUEST",Appartement,3.5,76 m2,2.,CHF 1'830.�,Immostreet
161,17/03/2017,Rent,Lausanne,CH. DU TRABANDAN 37 C,1006 Lausanne,"Cuisine : laboratoire, ouverte s/coin manger / Agencement : cuisini�re �lectrique, frigo, hotte ventilation, lave-vaisselle / Caract�ristiques : hall, cave . Orientation : OUEST",Appartement,3.5,76 m2,3.,CHF 1'870.�,Immostreet
162,17/03/2017,Rent,Lausanne,Ch. du Vanil 8,1006 Lausanne,"Appartement comprenant : hall, 2 chambres, s�jour, cuisine agenc�e, bains/WC. Quartier tranquille. Orient� plein Sud. Place de parc ext�rieure � disposition � Fr. 80.00. .",Appartement,3.5,70 m2,R.D.C.,CHF 1'900.�,Immostreet
163,17/03/2017,Rent,Lausanne,Chemin de Bonne-Esp�rance 7,1006 Lausanne,"quartier tranquille et proche de toutes les commodit�s, cuisine enti�rement agenc�e, salle de bains-WC, balcon.",Appartement,2.5,70 m2,4.,CHF 1'960.�,Immostreet
164,17/03/2017,Rent,Lausanne,Chemin de Bonne-Esp�rance 5,1006 Lausanne,"proche du magasin alimentaire COOP et des transports, cuisine agenc�e, salle de bains-WC, balcon.",Appartement,2.5,70 m2,4.,CHF 2'020.�,Immostreet
165,17/03/2017,Rent,Lausanne,Chemin de Beau-Rivage 6,1006 Lausanne,"Cet appartement de 59 m2 est id�alement situ� � deux pas du lac et compos� comme suit: Hall d'entr�e avec armoires murales, s�jour, cuisine partiellement agenc�e, chambre, salle de bains/WC Contact...",Appartement,2.5,59 m2,4.,CHF 2'260.�,Immostreet
166,17/03/2017,Rent,Lausanne,Avenue de l'Elys�e 23,1006 Lausanne,"Nouvelle construction de 2015, immeuble de standing au coeur du quartier de Montchoisi Appartement de 2.5 pi�ces au 2�me �tage Ce dernier est compos� de la fa�on suivante: Un hall d'entr�e Une cham...",Appartement,2.5,51 m2,2.,CHF 2'490.�,Immostreet
167,17/03/2017,Rent,Lausanne,Chemin du Closelet 6,1006 Lausanne,"Magnifique immeuble situ� � deux pas de la gare de Lausanne et 10 min � pieds d'Ouchy, proche de toutes commodit�s et des transports publics. Acc�s s�curis�, ascenseur. comprenant : 2 grandes chamb...",Appartement,3,95 m2,4.,CHF 2'750.�,Immostreet
168,17/03/2017,Rent,Lausanne,Avenue des Alpes 2Bis,1006 Lausanne,"APPARTEMENT MEUBLE ET ENTI�REMENT �QUIP� A LOUER Lausanne Logement de 3,5 pi�ces au 3�me �tage Immeuble id�alement situ� � proximit� imm�diate des commerces et des commodit�s. Belle superficie. Fin...",Appartement,3.5,,R.D.C.,CHF 2'800.�,Immostreet
169,17/03/2017,Rent,Lausanne,Juste Olivier 25,1006 Lausanne,"Bel appartement de 4.5 pi�ces id�alement situ� dans le quartier sous gare, � proximit� de toutes les commodit�s. L'appartement a �t� enti�rement r�nov� en 2013, il offre tout le charme de l'ancien ...",Appartement,4.5,131 m2,2.,CHF 3'200.�,Immostreet
170,17/03/2017,Rent,Lausanne,chemin du Trabandan 24,1006 Lausanne,"Appartement de standing de 4 pi�ces dans petit immeuble avec magnifique vue sur le lac, quartier calme comprenant : Hall, deux grandes chambres � coucher, cuisine enti�rement agenc�e, salle � mange...",Appartement,4,135 m2,1.,CHF 3'200.�,Immostreet
171,17/03/2017,Rent,Lausanne,Avenue de l'Elys�e 23,1006 Lausanne,"Nouvelle construction de 2015, immeuble de standing au coeur du quartier de Montchoisi Appartement de standing jouissant d'une excellente situation et luminosit�. Ce dernier est compos� de la fa�on...",Appartement,3.5,79 m2,1.,CHF 3'200.�,Immostreet
172,17/03/2017,Rent,Lausanne,Avenue Juste-Olivier,1006 Lausanne,"Magnifique appartement r�nov� avec beaucoup de cachet. Situ� au centre de ville de Lausanne au 4�me �tage d'un immeuble cossu, cet appartement de 4 1/2 pi�ces comprend : - Un hall spacieux - Une cu...",Appartement,4.5,101 m2,4.,CHF 3'760.�,Immostreet
173,17/03/2017,Rent,Lausanne,Avenue de Montchoisi 9,1006 Lausanne,Magnifique loft a louer sous gare � coter d'Ouchy. Tr�s belle surface et tr�s bon �tat avec cuisine contemporaine agenc�e. Une grande cave ainsi qu'une place de parc disponible. Immeuble avec gardien.,Loft,,150 m2,3.,CHF 3'769.�,Immostreet
174,17/03/2017,Rent,Lausanne,Chemin de Montolivet 1,1006 Lausanne,"Bel appartement de 4.5 pi�ces au 3�me �tage avec vue imprenable sur le lac et les alpes qui se compose comme suit : Un hall d'ent�e, une cuisine agenc�e et �quip�e ouverte sur un s�jour avec chemin...",Appartement,4.5,115 m2,,CHF 3'850.�,Immostreet
175,17/03/2017,Rent,Lausanne,Av. de Montchoisi 20A,1006 Lausanne,"Appartement de 4.5 pi�ces d'exception en attique, au coeur de Lausanne, se composant comme suit : Ascenseur privatif Terrasse panoramique Grand hall avec armoires Cuisine ouverte sur spacieux salon...",Appartement,4.5,124 m2,5.,CHF 3'870.�,Immostreet
176,17/03/2017,Rent,Lausanne,Av. Juste-Olivier 6,1006 Lausanne,"APPARTEMENT D'EXCEPTION Centre-ville, proche de toutes commodit�s, des transports publics et � 2 minutes de la gare CFF, splendide appartement de 3.5 pi�ces de haut standing, situ� dans les combles...",Appartement,3.5,120 m2,5.,CHF 4'100.�,Immostreet
177,17/03/2017,Rent,Lausanne,Avenue de l'Elys�e 23,1006 Lausanne,"Nouvelle construction de 2015, immeuble de standing au coeur du quartier de Montchoisi Bel appartement de standing 4.5 pi�ces au 3�me �tage Appartement comprenant: une cuisine enti�rement agenc�e o...",Appartement,4.5,97 m2,3.,CHF 4'210.�,Immostreet
178,17/03/2017,Rent,Lausanne,Avenue Antoine-Michel-Servan 12,1006 Lausanne,"Immeuble en copropri�t� situ� dans un quartier ""sous-gare"" calme. A quelques minutes � pied de la gare de Lausanne, du m�tro et d'Ouchy. Commodit�s � la porte. Comprenant : hall avec armoires mural...",Duplex,4.5,150 m2,R.D.C.,CHF 4'250.�,Immostreet
179,17/03/2017,Rent,Lausanne,Avenue des Alpes 22,1006 Lausanne,Petit immeuble r�sidentiel de 4 appartements au centre ville (Quartier sous-gare). Situ� dans quartier calme et verdoyant. Proches de toutes commodit�s. Gare et M2 � 10 minutes � pied. Comprenant: ...,Appartement,6.5,142 m2,1.,CHF 4'650.�,Immostreet
180,17/03/2017,Rent,Lausanne,Avenue des Alpes 22,1006 Lausanne,Petit immeuble r�sidentiel de 4 appartements au centre ville (quartier sous-gare). Situ� dans quartier calme et verdoyant. Proches de toutes commodit�s. Gare et M2 � 10 minutes � pied. Appartement ...,Appartement,6.5,142 m2,3.,CHF 4'970.�,Immostreet
181,17/03/2017,Rent,Lausanne,Avenue Tissot 16,1006 Lausanne,Bel appartement de 5 pi�ces au rez inf�rieur d'env. 175m2 comprenant : - hall habitable - s�jour avec chemin�e - 3 chambres - un Walking closet - une salle � manger - une cuisine enti�rement agenc�...,Appartement,5,175 m2,R.D.C.,CHF 5'460.�,Immostreet
182,17/03/2017,Rent,Lausanne,Chemin de Montolivet 35,1006 Lausanne,"Magnifique appartement situ� au 3�me et dernier �tage d�un immeuble d�un des beaux quartiers de Lausanne, id�alement situ� � seulement 5 min d�Ouchy. Il est entour� de verdure au calme, dans un imm...",Attique,7.5,208 m2,,CHF 5'850.�,Immostreet
183,17/03/2017,Rent,Lausanne,Chemin de Montolivet 35,1006 Lausanne,"Id�alement situ� dans un �crin de verdure au calme absolu tout en �tant � proximit� (soit � 5 minutes � pied) de toutes commodit�s, des transports, du lac, du parc du Denantou ainsi que de la prest...",Appartement,5.5,208 m2,3.,CHF 5'850.�,Immostreet
184,17/03/2017,Rent,Lausanne,,1006 Lausanne,"Dans un immeuble de la fin des ann�es 60, dans un quartier calme et proche des transports publics, cet appartement enti�rement r�nov� avec go�t en 2013 vous s�duira par ses volumes. Il se compose c...",Appartement,6.5,195 m2,,CHF 5'900.�,Immostreet
185,17/03/2017,Rent,Lausanne,Avenue Eglantine 7,1006 Lausanne,"Situ� dans un quartier privil�gi� de Rumine dans un magnifique b�timent avec beaucoup de cachet, bel appartement de 5.5 pi�ces de 180m2 habitables avec hauts plafonds et moulures. Il b�n�ficie d'un...",Appartement,5.5,180 m2,1.,CHF 5'900.�,Immostreet
186,17/03/2017,Rent,Lausanne,chemin de Montolivet,1006 Lausanne,"Magnifique appartement situ� dans un quartier r�sidentiel, proche du lac, du mus�e Olympique et du centre ville. Les coll�ges de Champittet, de Montchoisi et de l'Elys�e dans les proches alentours,...",Attique,7,200 m2,3.,CHF 6'200.�,Immostreet
187,17/03/2017,Rent,Lausanne,Chemin de Montolivet 35,1006 Lausanne,"Id�alement bien situ� � quelques minutes du Lac et proche de la ville de Lausanne. Ascenseur privatif, hall avec armoires murales, salon avec chemin�e et acc�s � la terrasse avec BBQ, salle � mange...",Attique,6.5,208 m2,3.,CHF 6'200.�,Immostreet
188,17/03/2017,Rent,Lausanne,Avenue Eglantine 5,1006 Lausanne,"Situ� dans un quartier privil�gier de Rumine, magnifique appartement de 6 pi�ces de 200m2 habitables enti�rement r�nov� en 2013 avec une terrasse. A deux pas des transports, � 5 minutes de la gare ...",Appartement,6,200 m2,2.,CHF 6'250.�,Immostreet
189,17/03/2017,Rent,Lausanne,Chemin Edouard-Sandoz 7/9,1006 Lausanne,"Cette propri�t� de ma�tre d'environ 740 m2 est implant�e sur une belle parcelle de plus de 7000 m2 et profite d'une situation privil�gi�e, � proximit� du centre ville de Lausanne et des Quais d'Ouc...",Maison individuelle,14,740 m2,,,Immostreet
190,17/03/2017,Rent,Lausanne,Chemin de Bonne-Esp�rance 20,1006 Lausanne,"Ce studio d'une surface d'environ 32m2, est situ� dans un quartier tranquille de Lausanne. Cet appartement vous offre: un s�jour une cuisine �quip�e (sans lave-vaisselle) une salle-de bains L'immeu...",Appartement,1,32 m2,3.,,Immostreet
191,17/03/2017,Rent,Lausanne,Avenue du Mont-d'Or 37,1007 Lausanne,"Chambre ind�pendante au rez inf�rieur avec un lavabo, r�chaud 2 plaques et frigo. Usage du WC commun sur l'�tage. Visites les mercredis de 15h � 18h.",Chambre,1,10 m2,R.D.C.,CHF 485.�,Immostreet
192,17/03/2017,Rent,Lausanne,Avenue du Mont-d'Or 35,1007 Lausanne,Chambre au rez-de-chauss�e avec lavabo. Visites les mercredis de 15h � 18h.,Chambre,1,13 m2,R.D.C.,CHF 495.�,Immostreet
193,17/03/2017,Rent,Lausanne,Avenue de Tivoli 24,1007 Lausanne,"logement comprenant hall, cuisine agenc�e, salle de bains-WC.",Appartement,1.5,44 m2,R.D.C.,CHF 1'190.�,Immostreet
194,17/03/2017,Rent,Lausanne,Chemin de la Tour-Grise 28,1007 Lausanne,"Studio enti�rement r�nov� compos� comme suit : hall (avec armoires encastr�es), s�jour, cuisine avec cuisini�re vitroc�ram, four, hotte, r�frig�rateur, douche-wc, terrasse privative, cave.",Appartement,1,33 m2,R.D.C.,CHF 1'190.�,Immostreet
195,17/03/2017,Rent,Lausanne,CH. DU-BOIS-DE-LA-FONTAINE  9,1007 Lausanne,"Cuisine : fen�tre, habitable ferm�e / Agencement : frigo, hotte ventilation, lave-vaisselle, vitroc�ram / Caract�ristiques : balcon, hall, cave . Orientation : EST, NORD",Appartement,2,53 m2,3.,CHF 1'335.�,Immostreet
196,17/03/2017,Rent,Lausanne,Chemin du Couchant 33,1007 Lausanne,"situ� dans un quartier calme et verdoyant, � c�t� du parc de la vall�e de la jeunesse, cuisine agnec�e, salle de bains-WC, balcon.",Appartement,2.5,59 m2,1.,CHF 1'375.�,Immostreet
197,17/03/2017,Rent,Lausanne,Bois-de-Vaux 11,1007 Lausanne,Laissez-vous s�duire pas ce logement pour ses qualit�s suivantes : � Hall � Salon avec acc�s au balcon � Chambre avec parquet � Salle de bains / WC � Cave � Ascenseur Disponible pour le 1 juin 2017...,Appartement,2,52 m2,1.,CHF 1'400.�,Immostreet
198,17/03/2017,Rent,Lausanne,Chemin du Mont-Tendre 8 ter,1007 Lausanne,"Immeuble dans quartier calme et proche de la gare. Petit loft enti�rement r�nov�. Logement comprenant une grande pi�ce � vivre, une cuisine agenc�e et une salle de douches.",Studio,1,29 m2,1.,CHF 1'480.�,Immostreet
199,17/03/2017,Rent,Lausanne,Passage Fr.-Bocion 5,1007 Lausanne,"Id�alement situ�, l'immeuble est proche de toutes commodit�s et le m�tro est � deux pas. A louer au 16.03.2017, bail de dur�e max jusqu'au 31.01.2021. Possibilit� de r�silier chaque mois avec un pr...",Appartement,2,51 m2,1.,CHF 1'520.�,Immostreet
200,17/03/2017,Rent,Lausanne,Route de Chavannes,1007 Lausanne,"Appartement de 2.5 pi�ces au rez-de-chauss�e comprenant hall d'entr�e, s�jour avec terrasse, 1 chambre � coucher, cuisine semi agenc�e, salle de bains/WC.",Appartement,2.5,58 m2,R.D.C.,CHF 1'540.�,Immostreet
201,17/03/2017,Rent,Lausanne,Ch. de Contigny 5,1007 Lausanne,"Joli 2 pi�ces comprenant : hall d'entr�e, cuisine enti�rement agenc�e (vitroc�ram, frigo avec bacs cong�lateur), salle de bains/WC, salon et une chambre. Parquet flottant A louer pour le 1 avril 20...",Appartement,2,42 m2,1.,CHF 1'560.�,Immostreet
202,17/03/2017,Rent,Lausanne,Bois-de-Vaux 11,1007 Lausanne,"Spacieux logement comprenant, hall avec armoire, salle de bains, WC s�par�, 2 chambres, salon, deux balcons. R�fection peinture + parquet avant relocation ! A louer pour le 1 avril 2017. N'h�sitez ...",Appartement,3.5,72 m2,1.,CHF 1'690.�,Immostreet
203,17/03/2017,Rent,Lausanne,Av. Marc-Dufour 44,1007 Lausanne,"Entr�e, grand d�gagement meublable avec fen�tre, cuisine ferm�e, salle de bains-WC, deux balcons.",Appartement,2.5,58 m2,2.,CHF 1'690.�,Immostreet
204,17/03/2017,Rent,Lausanne,Ch. de Montelly 16,1007 Lausanne,"Appartement avec poste de conciergerie. Salaire brut: Fr. 550.00. Entr�e, cuisine ferm�e, frigo, cuisini�re �lectrique, lave-vaisselle, hotte, salle de bains-WC.",Appartement,3,66 m2,1.,CHF 1'700.�,Immostreet
205,17/03/2017,Rent,Lausanne,Avenue de Tivoli 24,1007 Lausanne,"compos� d'un hall, s�jour, cuisine, chambre, salle de bains-WC, balcon.",Appartement,2.5,62 m2,4.,CHF 1'710.�,Immostreet
206,17/03/2017,Rent,Lausanne,Ch. des Matines 7,1007 Lausanne,"Entr�e, cuisine ferm�e, frigo, salle de bains avec bidet, WC s�par�, balcon. R�serv� : un contrat est en cours pour cet objet.",Appartement,4,72 m2,2.,CHF 1'770.�,Immostreet
207,17/03/2017,Rent,Lausanne,Chemin de la Prairie 5A,1007 Lausanne,"A louer pour le 1er mars 2017 ou � convenir, bel appartement prot�g� de 3.5 pi�ces au 1er �tage, ascenseur, balcon. Loyer mensuel CHF 1'800.00 charges comprises. Ce logement s'adresse aux personnes...",Appartement,3.5,81 m2,1.,CHF 1'800.�,Immostreet
208,17/03/2017,Rent,Lausanne,Avenue de Tivoli 70,1007 Lausanne,"Appartement de 3 pi�ces au 3�me �tage comprenant une entr�e, une cuisine agenc�e, un s�jour, une salle de bains, un WC s�par�, deux chambres � coucher et un balcon.",Appartement,,79 m2,,CHF 1'810.�,Immostreet
209,17/03/2017,Rent,Lausanne,Ch. de la Bateli�re 3,1007 Lausanne,"Cuisine agenc�e, salle de bains, WC s�par�, balcon.",Appartement,3.5,70 m2,3.,CHF 1'830.�,Immostreet
210,17/03/2017,Rent,Lausanne,Ch. de Contigny 5,1007 Lausanne,"Spacieux logement comprenant, hall, salle de bains-WC, cuisine agenc�e (sauf lave-vaisselle), 3 chambres, salon + 1x chambre donnant sur le balcon. A louer pour le 16 avril 2017. Visite pr�vue le l...",Appartement,4,73 m2,2.,CHF 1'860.�,Immostreet
211,17/03/2017,Rent,Lausanne,Chemin du Suchet 5,1007 Lausanne,"Magnifique appartement meubl�d'environ 40m2 et sa terrasse de 18m2. Id�alement situ� dans le quartier tr�s pris� des Suchets! Proche du Parc de Milan, de l'�cole Montriond et de deux points de Mob...",Appartement,1.5,40 m2,R.D.C.,CHF 1'870.�,Immostreet
212,17/03/2017,Rent,Lausanne,Avenue de la Harpe 10,1007 Lausanne,"Entre la gare de Lausanne et Ouchy dans un immeuble de caract�re, charmant appartement meubl� de 2,5 pi�ces avec hauts plafonds comprenant : entr�e avec armoire, chambre � coucher, salon avec chemi...",Appartement,2.5,,1.,CHF 1'900.�,Immostreet
213,17/03/2017,Rent,Lausanne,Av. de S�velin 2B,1007 Lausanne,"Entr�e, cuisine ouverte avec coin � manger, frigo, cuisini�re �lectrique, lave-vaisselle, salle de bains, WC s�par�, balcon.",Appartement,3,77 m2,5.,CHF 1'965.�,Immostreet
214,17/03/2017,Rent,Lausanne,Chemin de Bellerive 21,1007 Lausanne,"dans bel immeuble situ� dans un quartier calme, proche du lac et de toutes commodit�s, cuisine agenc�e ouverte sur un grand s�jour avec baie vitr�e, une chambre � coucher, douche-WC.",Appartement,2.5,79 m2,4.,CHF 2'150.�,Immostreet
215,17/03/2017,Rent,Lausanne,Rue du Cr�t 4,1007 Lausanne,"Comprenant: hall d'entr�e, cuisine agenc�e, salon/salle-�-manger, W.C. s�par�s, 2 chambres salle-de-bains/W.C., balcon.",Appartement,3.5,88 m2,2.,CHF 2'260.�,Immostreet
216,17/03/2017,Rent,Lausanne,Avenue de Cour 17,1007 Lausanne,"Appartement meubl� de 3,5 pi�ces au 2�me �tage b�n�ficiant d'une cuisine agenc�e, de deux chambres, un s�jour, une salle de douche et un WC s�par�.",Appartement,3.5,,2.,CHF 2'450.�,Immostreet
217,17/03/2017,Rent,Lausanne,Av. de Cour 155,1007 Lausanne,"Dans immeuble r�sidentiel et id�alement situ� � proximit� du centre ville, du lac, des transports publics et des axes autoroutiers, tr�s bel appartement de 2.5 pi�ces, d'une surface d'env. 81 m2, b...",Appartement,2.5,81 m2,7.,CHF 2'510.�,Immostreet
218,17/03/2017,Rent,Lausanne,"59, avenue du Mont-d'Or",1007 Lausanne,"Situ� dans un quartier r�sidentiel proche de toutes les commodit�s ( commerces, transports...), ce spacieux appartement de 2.5 pi�ces d'environ 70m2 se compose comme suit : Cuisine am�nag�e et �qui...",Appartement,2.5,80 m2,1.,CHF 2'700.�,Immostreet
219,17/03/2017,Rent,Lausanne,Avenue de Tivoli 56,1007 Lausanne,"Appartement de 4 pi�ces au 3�me �tage comprenant : Hall, Cuisine agenc�e, S�jour, 3 chambres, Salle de bains, WC S�par�, Loggia Disponibilit� : 15 avril 2017 Contact pour visite : Mme Bistuer, loca...",Appartement,4,107 m2,3.,CHF 2'710.�,Immostreet
220,17/03/2017,Rent,Lausanne,Avenue de Montoie 37,1007 Lausanne,Proche de toutes commodit�s. Appartement de 4.5 pi�ces au 12�me �tage avec une magnifique vue sur le lac et les montagnes. Cet appartement est compos� de: Hall d'entr�e avec penderie et armoires mu...,Appartement,4,116 m2,12.,CHF 2'900.�,Immostreet
221,17/03/2017,Rent,Lausanne,Avenue des Figuiers 20,1007 Lausanne,"Proche de toutes commodit�s dans quartier agr�able de Lausanne, ce 3.5 pi�ces r�cemment r�nov� � une vue �poustouflante sur le lac et les montagnes, il se compose comme suit : - Hall d'entr�e avec ...",Appartement,3.5,95 m2,8.,CHF 3'200.�,Immostreet
222,17/03/2017,Rent,Lausanne,Chemin des Mouettes 24,1007 Lausanne,"Ce charmant duplex, r�cemment r�nov�, comporte 2,5 pi�ces, d'environ 120 m2 et prend place au rez-de-chauss�e d'un petit immeuble, sis dans un quartier r�sidentiel de Lausanne-Ouchy, au chemin des ...",Duplex,2.5,120 m2,,CHF 3'400.�,Immostreet
223,17/03/2017,Rent,Lausanne,Chemin de la Bateli�re 4,1007 Lausanne,"Grand appartement de 4.5 pi�ces situ� au 1er �tage avec magnifique vue sur le lac. Constitu� d'un s�jour luminueux avec chemin�e, une cuisine enti�rement �quip�e, 3 chambre, une salle de bain (avec...",Appartement,4.5,,1.,CHF 3'450.�,Immostreet
224,17/03/2017,Rent,Lausanne,Mont d'Or 59,1007 Lausanne,"Bel appartement contemporain de 4.5 pi�ces avec finitions de standing, armoires murales, parquet, buanderie priv�e, grand balcon dans le quartier tr�s pris� � sous-gare �. Id�ale pour famille avec ...",Appartement,4.5,,,CHF 3'800.�,Immostreet
225,17/03/2017,Rent,Lausanne,Avenue des Figuiers 13,1007 Lausanne,"Immeuble proche de toutes commodit�s et � deux pas du bord du lac. Grand appartement de 4.5 pi�ces au 4�me �tage, comprenant un hall, une cuisine enti�rement agenc�e et �quip�e ouverte sur un vaste...",Appartement,4.5,102 m2,4.,CHF 4'061.�,Immostreet
226,17/03/2017,Rent,Lausanne,Avenue de la Harpe 49,1007 Lausanne,"Appartement de 2.5 pi�ces au 3�me �tage compos� d'un hall avec armoire murale, un s�jour, une cuisine ouverte agenc�e, une chambre � coucher d'env. 20 m� avec salle de douches/WC et dressing attena...",Appartement,2.5,92 m2,3.,CHF 4'100.�,Immostreet
227,17/03/2017,Rent,Lausanne,Chemin des Mouettes 22,1007 Lausanne,"Splendide appartement de 3.5 pi�ces de tr�s grand standing distribu� en Duplex d'environ 106m2 au 1er �tage d'un immeuble proche de toutes les commodit�s, des commerces, des bus, du m�tro et � quel...",Appartement,3.5,106 m2,1.,CHF 4'900.�,Immostreet
228,17/03/2017,Rent,Lausanne,Av. des Bains 8,1007 Lausanne,"Magnifique appartement dans nouvelle promotion d'une copropri�t� de haut standing, jouissant d'une tr�s belle vue sur tout le bassin l�manique, � proximit� de toutes commodit�s. . 3 minutes � pied ...",Appartement,5.5,124 m2,3.,CHF 5'250.�,Immostreet
229,17/03/2017,Rent,Lausanne,Avenue de Grammont 11,1007 Lausanne,"Dans immeuble ancien de standing avec beaucoup de charme, hauts plafonds, calme et verdoyant, � deux pas du M2, proche de la gare et du centre ville, magnifique appartement en duplex de 8.5 pi�ces,...",Appartement,8.5,382 m2,R.D.C.,CHF 7'530.�,Immostreet
230,17/03/2017,Rent,Lausanne,Place de Milan 18,1007 Lausanne,Grande maison de charme id�alement situ�e � proximit� de toutes les commodit�s. Cette belle demeure jouie d'un emplacement de choix et dispose d'un beau jardin et une belle piscine privative. Enti�...,Maison individuelle,9,285 m2,,CHF 8'500.�,Immostreet
231,17/03/2017,Rent,Lausanne,Avenue de Milan 18,1007 Lausanne,"Situ� dans le quartier tr�s pris� de sous-gare, ce bien de charme �rig� en 1904 a �t� enti�rement r�nov� et b�n�ficie d'un beau jardin intime agr�ment� d'une piscine ext�rieure. La cour offre la po...",Villa,9,280 m2,,CHF 8'500.�,Immostreet
232,17/03/2017,Rent,Lausanne,Avenue de la Harpe 14,1007 Lausanne,"Appartement 2 pi�ces d'environ 55 m2 au rez-de-chauss�e d'un immeuble class� historique, ce logement situ� dans le quartier sous gare, saura vous s�duire par son vieux parquet et ses hauts plafonds...",Appartement,2,,R.D.C.,,Immostreet
233,17/03/2017,Rent,Lausanne,Avenue de Valmont 14,1010 Lausanne,"A proximit� des axes autoroutiers, proche des transports publics et de toutes commodit�s. Id�al pour un �tudiant. Appartement comprenant : - Une pi�ce - Cuisine avec frigo - Salle de bains/WC - Cav...",Studio,1,37 m2,1.,CHF 920.�,Immostreet
234,17/03/2017,Rent,Lausanne,CH. DES LIBELLULES 8,1010 Lausanne,"Cuisine : ouverte s/coin manger / Agencement : cuisini�re �lectrique, frigo, hotte ventilation, lave-vaisselle / Caract�ristiques : balcon, cave . Orientation : EST, OUEST",Appartement,2.5,55 m2,2.,CHF 1'290.�,Immostreet
235,17/03/2017,Rent,Lausanne,Route de la Feuill�re 21,1010 Lausanne,"Magnifique immeuble Minergie de 2014, construit en qualit� PPE. Spacieux logement de standing avec balcon. Logement compos� d'une cuisine agenc�e avec lave-vaisselle ouverte sur la pi�ce principale...",Appartement,1.5,34 m2,1.,CHF 1'355.�,Immostreet
236,17/03/2017,Rent,Lausanne,Route de Berne 1,1010 Lausanne,"Immeuble proche du CHUV, de toutes les commodit�s et aux portes du M2. Bel appartement 2 pi�ces - Bail � dur�e d�termin�e jusqu'au 31.05.2019 Comprenant: - Une cuisine - Une chambre - Un s�jour ave...",Appartement,2,44 m2,8.,CHF 1'360.�,Immostreet
237,17/03/2017,Rent,Lausanne,Chemin de Boissonnet 96,1010 Lausanne,"hall avec armoires murales, 1 chambre, cuisine agenc�e avec hotte, four, frigo, salle de douche-WC, cave.",Studio,1,29 m2,2.,CHF 1'370.�,Immostreet
238,17/03/2017,Rent,Lausanne,Route d'Oron 14C,1010 Lausanne,"Quartier de la Sallaz, bus � la porte. Magasins, m�tro et toutes autres commodit�s � proximit� imm�diate. Acc�s autoroutier facile. Appartement de 2 pi�ces au 2�me �tage. Comprenant: une cuisine ag...",Appartement,2,49 m2,2.,CHF 1'430.�,Immostreet
239,17/03/2017,Rent,Lausanne,Rte de Berne 111,1010 Lausanne,"Spacieux appartement de 1 pi�ce se situant au rez-de-chauss�e d'un immeuble neuf, de construction Minergie. Il se compose d'une entr�e avec une armoire murale, d'une cuisine agenc�e, d'une grande p...",Appartement,1,44 m2,R.D.C.,CHF 1'450.�,Immostreet
240,17/03/2017,Rent,Lausanne,Route de la feuillere 21,1010 Lausanne,cuisine �quip�e + lave vaisselle ... CONTACT OLIVIER VAN BELLINGEN� email: oliviervanbellingen@hotmail.com tel: +41798496978,Appartement,1.5,35 m2,1.,CHF 1'455.�,Immostreet
241,17/03/2017,Rent,Lausanne,Chemin des Lys 14,1010 Lausanne,"Immeuble id�alement situ�, dans un secteur calme et proche de toutes commodit�s. Transports publics � proximit�. comprenant : hall avec armoire, cuisine ferm�e, agenc�e-�quip�e, grand salon, une ch...",Appartement,2,52 m2,R.D.C.,CHF 1'490.�,Immostreet
242,17/03/2017,Rent,Lausanne,Chemin de Boissonnet 96,1010 Lausanne,"hall avec armoires murales, 1 chambre, cuisine agenc�e avec hotte, four, frigo, salle de douche-WC, cave.",Studio,1,20 m2,R.D.C.,CHF 1'490.�,Immostreet
243,17/03/2017,Rent,Lausanne,Chemin de Boissonnet 96,1010 Lausanne,"hall avec armoires murales, 1 chambre, cuisine agenc�e avec hotte, four, frigo, salle de douche-WC, cave.",Studio,1,20 m2,1.,CHF 1'500.�,Immostreet
244,17/03/2017,Rent,Lausanne,Champ-Rond53,1010 Lausanne,"Joli studio dans les environs du Parc de Sauvabelin, avec jardin. Avec cuisine �quip�e, salle de douche/WC. Dans un quartier tr�s calme, pr�s de toutes commodit�s.",Appartement,1,,R.D.C.,CHF 1'500.�,Immostreet
245,17/03/2017,Rent,Lausanne,Route de Berne 24,1010 Lausanne,"Immeuble proche des transports en communs et de toutes les commodit�s. Bel appartement lumineux Comprenant : - Cuisine agenc�e sans lave-vaisselle ouverte sur la salle � manger, - Un grand s�jour, ...",Appartement,2.5,54 m2,2.,CHF 1'520.�,Immostreet
246,17/03/2017,Rent,Lausanne,Chemin de Boissonnet 96,1010 Lausanne,"hall, 1 chambre, cuisine agenc�e avec hotte, four, frigo, salle de douche-WC, cave.",Studio,1,21 m2,R.D.C.,CHF 1'530.�,Immostreet
247,17/03/2017,Rent,Lausanne,Chemin de Boissonnet 96,1010 Lausanne,"hall avec armoires murales, 1 chambre, cuisine agenc�e avec hotte, four, frigo, salle de douche-WC, cave.",Studio,1,21 m2,1.,CHF 1'530.�,Immostreet
248,17/03/2017,Rent,Lausanne,Chemin de Boissonnet 96,1010 Lausanne,"hall, 1 chambre, cuisine agenc�e avec hotte, four, frigo, salle de douche-WC, cave. Visite : du lundi au jeudi � partir de 18h30.",Studio,1,20 m2,R.D.C.,CHF 1'530.�,Immostreet
249,17/03/2017,Rent,Lausanne,Chemin de Boissonnet 96,1010 Lausanne,"hall avec armoires murales, 1 chambre, cuisine agenc�e avec hotte, four, frigo, salle de douche-WC, cave.",Studio,1,21 m2,R.D.C.,CHF 1'550.�,Immostreet
250,17/03/2017,Rent,Lausanne,Route de Berne 111,1010 Lausanne,"Magnifique appartement, situ� dans une tr�s belle construction avec de belles finitions disposant d�un grand volume. Plac� id�alement sur la ligne du M2 (Arr�t fourmi en bas de l'immeuble) et � l�e...",Appartement,1.5,47 m2,2.,CHF 1'590.�,Immostreet
251,17/03/2017,Rent,Lausanne,Chemin de Boissonnet 96B,1010 Lausanne,"A louer � Lausanne, appartement de 2,5 pi�ces, au 2�me �tage, tr�s lumineux, avec vue d�gag�e et grand balcon-terrasse, comprenant: 1 chambre � coucher, 1 s�jour, une cuisine enti�rement agenc�e, u...",Appartement,2.5,,2.,CHF 1'660.�,Immostreet
252,17/03/2017,Rent,Lausanne,Chemin de Boissonnet 96,1010 Lausanne,"hall, 1 chambre, cuisine agenc�e avec hotte, four, frigo, salle de douche-WC, cave.",Studio,1,29 m2,1.,CHF 1'690.�,Immostreet
253,17/03/2017,Rent,Lausanne,Avenue des Boveresses 28,1010 Lausanne,"Dans le quartier des Boveresses, � proximit� des commodit�s et des acc�s autoroutiers, appartement de 3,5 pi�ces au rez-de-chauss�e comprenant: un hall d'entr�e, une cuisine �quip�e, un s�jour, deu...",Appartement,3.5,,R.D.C.,CHF 1'750.�,Immostreet
254,17/03/2017,Rent,Lausanne,Route de Berne 1,1010 Lausanne,"Immeuble proche du CHUV, de toutes les commodit�s et aux portes du M2. Bel appartement de 3.5 pi�ces - Bail � dur�e d�termin�e jusqu'au 31.05.2019. Compenant : - Hall avec armoires muralles, - Une ...",Appartement,3.5,72 m2,5.,CHF 1'800.�,Immostreet
255,17/03/2017,Rent,Lausanne,Chemin de B�r�e 14A,1010 Lausanne,"Nouvelle construction minergie ECO � Lausanne. Appartement r�serv� pour les s�niors ou les personnes � mobilit� r�duite. Quartier desservi par les transports publics : M2, arr�t Fourmi et acc�s pro...",Appartement,2.5,56 m2,3.,CHF 1'800.�,Immostreet
256,17/03/2017,Rent,Lausanne,Chemin de Champ-Rond 19,1010 Lausanne,"Quartier agr�able et verdoyant situ� � 8 minutes � pied de la Sallaz avec les commerces et le m�tro pour le centre ville. Acc�s autoroutier facile. Appartement de 3 pi�ces au rez-de-chauss�e, compr...",Appartement,3,76 m2,R.D.C.,CHF 1'810.�,Immostreet
257,17/03/2017,Rent,Lausanne,Avenue de Valmont 14,1010 Lausanne,"A proximit� des axes autoroutiers, proche des transports publics et de toutes commodit�s. Appartement r�cemment r�nov� Cet appartement est compos� de: Hall d'entr�e, cuisine agenc�e, s�jour, deux c...",Appartement,3,72 m2,8.,CHF 1'835.�,Immostreet
258,17/03/2017,Rent,Lausanne,Route de Berne 111,1010 Lausanne,"Lumineux appartement de 2,5 pi�ces au 5�me �tage am�nag� avec des mat�riaux de grande qualit� dans immeuble neuf r�pondant aux normes Minergie et id�alement situ� sur la ligne du m�tro M2 et d'une ...",Appartement,2.5,,5.,CHF 1'860.�,Immostreet
259,17/03/2017,Rent,Lausanne,Chemin de la B�r�e 12B,1010 Lausanne,"""R�sidence Les Fourmis"". Nouvelles constructions Minergie situ�es dans le quartier de la Sallaz, � deux pas de l'arr�t du M2 et des acc�s autoroutiers. Appartement spacieux. Comprenant : - Un s�jou...",Appartement,2.5,62 m2,3.,CHF 1'880.�,Immostreet
260,17/03/2017,Rent,Lausanne,111 Route de Berne,1010 Lausanne,"Au sein d'un immeuble neuf Minergie de standing, cet appartement de 2.5 pi�ces dispose d'une surface de 65 m2. L'appartement donne sur l'avenue Crousaz. L'immeuble se situe � proximit� imm�diate de...",Appartement,2.5,65 m2,2.,CHF 1'890.�,Immostreet
261,17/03/2017,Rent,Lausanne,Chemin de Boissonnet 96,1010 Lausanne,"hall, s�jour, 1 chambre, cuisine agenc�e avec hotte, four, frigo, salle de douche-WC, cave.",Appartement,2.5,46 m2,R.D.C.,CHF 1'960.�,Immostreet
262,17/03/2017,Rent,Lausanne,Chemin de Boissonnet 96,1010 Lausanne,"hall, s�jour, 1 chambre, cuisine agenc�e avec hotte, four, frigo, salle de douche-WC, cave. Visites : la semaine apr�s 19h.",Appartement,2.5,46 m2,1.,CHF 1'970.�,Immostreet
263,17/03/2017,Rent,Lausanne,Avenue Victor-Ruffy 31,1010 Lausanne,"Immeuble situ� dans un quartier calme � proximit� des transports publics, �coles et commerces. Id�al pour couple retrait�. Appartement 4 pi�ces au 2�me �tage compos� d'un hall d'entr�e, une cuisine...",Appartement,4,80 m2,2.,CHF 2'075.�,Immostreet
264,17/03/2017,Rent,Lausanne,Route de Berne 111,1010 Lausanne,"Magnifique appartement, situ� dans une tr�s belle construction avec de belles finitions disposant d�un grand volume. Plac� id�alement sur la ligne du M2 (Arr�t fourmi en bas de l'immeuble) et � l�e...",Appartement,1.5,47 m2,2.,CHF 2'250.�,Immostreet
265,17/03/2017,Rent,Lausanne,Route de la Feuill�re 13,1010 Lausanne,"Situ� dans les hauts de Lausanne, ce complexe b�n�ficie d'une situation id�ale proche de toutes commodit�s: Poste, banque et commerces se trouvent � moins de 500 m�tres. Par ailleurs, les �coles de...",Appartement,3.5,84 m2,R.D.C.,CHF 2'295.�,Immostreet
266,17/03/2017,Rent,Lausanne,Route d' Oron 11,1010 Lausanne,"Proche de toutes commodit�s et des commerces, quartier de la Sallaz. Le logement se compose d'un grand s�jour lumineux, cuisine s�par�e et �quip�e, d'une salle de bains, WC s�par� et de deux belles...",Appartement,3.5,86 m2,4.,CHF 2'335.�,Immostreet
267,17/03/2017,Rent,Lausanne,Route d' Oron 11,1010 Lausanne,"Proche de toutes commodit�s et des commerces, quartier de la Sallaz. Le logement se compose d'un grand s�jour lumineux, cuisine s�par�e et �quip�e, d'une salle de bains, WC s�par� et de deux belles...",Appartement,3.5,86 m2,4.,CHF 2'335.�,Immostreet
268,17/03/2017,Rent,Lausanne,Route de la Feuill�re15,1010 Lausanne,"Situ� dans les hauts de Lausanne, ce complexe b�n�ficie d'une situation id�ale proche de toutes commodit�s: Poste, banque et commerces se trouvent � moins de 500 m�tres. Par ailleurs, les �coles de...",Appartement,3.5,89 m2,R.D.C.,CHF 2'395.�,Immostreet
269,17/03/2017,Rent,Lausanne,Avenue des Boveresses 4,1010 Lausanne,"hall avec armoires murales, cuisine agenc�e et partiellement ouverte sur la salle � manger, s�jour attenant, 2 grandes chambres � coucher, salle de bains, WC s�par�, balcon, place de parc ext�rieur...",Appartement,4.5,108 m2,1.,CHF 2'450.�,Immostreet
270,17/03/2017,Rent,Lausanne,Rte de la Feuill�re 25,1010 Lausanne,"Ce bel appartement se situe au rez-de-chauss�e d'un petit immeuble Minergie, dans un quartier proche de la Sallaz et du M2. Il dispose de belles prestations, de finitions de choix et se d�compose c...",Appartement,3.5,85 m2,R.D.C.,CHF 2'480.�,Immostreet
271,17/03/2017,Rent,Lausanne,Champ-Rond 42,1010 Lausanne,"Enti�rement r�nov�, cuisine agenc�e, deux salles d'eau, douche et bain, 1 living, 2 chambres, 1 balcon, vue d�gag�e, situation calme proche bus et M2 La Sallaz. CHF 2'670.00 charges comprises. Fair...",Appartement,3.5,85 m2,1.,CHF 2'520.�,Immostreet
272,17/03/2017,Rent,Lausanne,Route de la Feuill�re 13,1010 Lausanne,"Appartement de 3.5 pi�ces au rez-de-chauss�e Situ� dans les hauts de Lausanne, ce complexe b�n�ficie d'une situation id�ale proche de toutes commodit�s: Poste, banque et commerces se trouvent � moi...",Appartement,3.5,89 m2,R.D.C.,CHF 2'545.�,Immostreet
273,17/03/2017,Rent,Lausanne,111 Route de Berne,1010 Lausanne,"Au sein d'un immeuble neuf Minergie de standing, cet appartement de 3.5 pi�ces dispose d'une surface de 93 m2 + 11 m2 de balcon. Id�alement situ� l'immeuble est � proximit� imm�diate des axes autor...",Appartement,3.5,93 m2,3.,CHF 2'550.�,Immostreet
274,17/03/2017,Rent,Lausanne,Av. des Boveresses 4,1010 Lausanne,"Situ� dans un quartier calme et proche de toutes commodit�s, cet appartement lumineux et enti�rement r�nov� en 2015, se compose comme suit : - S�jour avec acc�s balcon - Coin repas - Cuisine enti�r...",Appartement,4,108 m2,3.,CHF 2'600.�,Immostreet
275,17/03/2017,Rent,Lausanne,,1010 Lausanne,"A louer Magnifique appartement de standing de 3.5 pi�ces - 100 m2 Enti�rement r�nov� en 2006 et rafraichi en 2014 Proche de toute commodit�s (M2, Commerces, Poste, Pharmacie�) � Quartier Sallaz Cui...",Appartement,3.5,103 m2,,CHF 2'750.�,Immostreet
276,17/03/2017,Rent,Lausanne,Av. Des Boveresses 76,1010 Lausanne,Quartier familial aux portes de la ville! Ce lumineux logement se compose ainsi: Un grand hall d'entr�e Une cuisine ouverte sur la pi�ce � vivre Trois chambres dont une suite parentale b�n�ficiant ...,Appartement,4.5,109 m2,3.,CHF 2'827.�,Immostreet
277,17/03/2017,Rent,Lausanne,Boveresses 64 � 92,1010 Lausanne,"Dans un quartier familial aux portes de la ville et avec une garderie sur le site, nous vous proposons des 4.5 pi�ces de 107m2 et plus, qui se composent ainsi : Un grand hall d'entr�e (avec armoire...",Appartement,4.5,110 m2,1.,CHF 2'840.�,Immostreet
278,17/03/2017,Rent,Lausanne,Av. des Boveresses 68,1010 Lausanne,"Dans un immeuble de caract�re situ� dans un quartier calme et verdoyant, ce logement situ� au 3�me �tage saura vous s�duire pour ses qualit�s suivantes : Spacieux s�jour Armoires murales Cuisine ag...",Appartement,4.5,107 m2,3.,CHF 2'840.�,Immostreet
279,17/03/2017,Rent,Lausanne,Av. des Boveresses 68,1010 Lausanne,"Dans un immeuble de caract�re situ� dans un quartier calme et verdoyant, ce logement am�nag� au 2�me �tage saura vous s�duire pour ses qualit�s suivantes : 1 spacieux s�jour 3 chambres dont une ave...",Appartement,4.5,112 m2,2.,CHF 2'845.�,Immostreet
280,17/03/2017,Rent,Lausanne,Route de la Feuill�re 27,1010 Lausanne,"Situ� dans les hauts de Lausanne, ce complexe b�n�ficie d'une situation id�ale proche de toutes commodit�s: Poste, banque et commerces se trouvent � moins de 500 m�tres. Par ailleurs, les �coles de...",Appartement,4.5,104 m2,R.D.C.,CHF 2'860.�,Immostreet
281,17/03/2017,Rent,Lausanne,Av. Des Boveresses 66,1010 Lausanne,Ce spacieux 4.5 pi�ces de 112m2 saura vous s�duire non seulement par son am�nagement mais aussi pour ses qualit�s : Spacieux s�jour Armoires murales Cuisine agenc�e ouverte sur une grande pi�ce � v...,Appartement,4.5,112 m2,3.,CHF 2'915.�,Immostreet
282,17/03/2017,Rent,Lausanne,Route de la Feuill�re 19,1010 Lausanne,"Magnifique immeuble Minergie de 2014, construit en qualit� PPE. Superbe logement sur les hauts de la ville avec le confort moderne. Logement compos� d'un grand hall avec 4 armoires intergr�es, d'un...",Appartement,4.5,124 m2,1.,CHF 2'955.�,Immostreet
283,17/03/2017,Rent,Lausanne,Route de la Feuill�re 21,1010 Lausanne,"Magnifique immeuble Minergie de 2014, construit en qualit� PPE. Superbe logement sur les hauts de la ville avec le confort moderne. Logement compos� d'un grand hall avec 4 armoires intergr�es, d'un...",Appartement,4.5,124 m2,1.,CHF 2'955.�,Immostreet
284,17/03/2017,Rent,Lausanne,Avenue de Valmont 20,1010 Lausanne,"Ce logement traversant de 6 pi�ces, situ� au 3�me �tage de l'immeuble, est �quip� d'une cuisine ouverte et agenc�e (plan de travail en granit, cuisini�re vitroc�ram, four, lave-vaisselle, ...), de ...",Duplex,6,154 m2,,CHF 2'990.�,Immostreet
285,17/03/2017,Rent,Lausanne,Av. Des Boveresses 86,1010 Lausanne,"Dans un immeuble de caract�re situ� dans un quartier calme et verdoyant, ce logement saura vous s�duire pour ses qualit�s suivantes : Spacieux s�jour Armoires murales Cuisine agenc�e ouverte sur un...",Appartement,4.5,118 m2,1.,CHF 3'005.�,Immostreet
286,17/03/2017,Rent,Lausanne,Route de la Feuill�re 15,1010 Lausanne,"Situ� dans les hauts de Lausanne, ce complexe b�n�ficie d'une situation id�ale proche de toutes commodit�s: Poste, banque et commerces se trouvent � moins de 500 m�tres. Par ailleurs, les �coles de...",Appartement,3.5,98 m2,2.,CHF 3'005.�,Immostreet
287,17/03/2017,Rent,Lausanne,Chemin de B�r�e 18B,1010 Lausanne,"Nouvelle construction minergie ECO � Lausanne Quartier desservi par les transports publics : M2, arr�t Fourmi et acc�s pr�s de l�autoroute Appartement de 4,5 pi�ces en attique comprenant : hall d�e...",Appartement,4.5,95 m2,4.,CHF 3'020.�,Immostreet
288,17/03/2017,Rent,Lausanne,Av. des Boveresses 78,1010 Lausanne,"Dans un quartier familial aux portes de la ville, ce lumineux logement se compose ainsi: Un grand hall d'entr�e (avec armoires murales) Une cuisine ouverte sur la pi�ce � vivre Un grand s�jour Troi...",Appartement,4.5,118 m2,3.,CHF 3'025.�,Immostreet
289,17/03/2017,Rent,Lausanne,Avenue de Beaumont 82,1010 Lausanne,"Immeuble contemporain situ� entre le charmant quartier de Chailly et de La Sallaz � Lausanne. Celui-ci est � proximit� direct de l'autoroute, du m�tro (M2) et de toutes les commodit�s. Appartement ...",Appartement,3.5,89 m2,4.,CHF 3'029.�,Immostreet
290,17/03/2017,Rent,Lausanne,route de Berne 111,1010 Lausanne,"NOUVELLE CONSTRUCTION CERTIFIEE MINERGIE! Magnifique appartement de 4.5 pi�ces en attique avec vue sur le lac comprenant : hall, s�jour, trois chambres � coucher avec parquet, cuisine enti�rement a...",Appartement,4.5,100 m2,6.,CHF 3'200.�,Immostreet
291,17/03/2017,Rent,Lausanne,Avenue de Valmont 12,1010 Lausanne,"A proximit� des axes autoroutiers, proche des transports publics et de toutes commodit�s. Spacieux appartement en duplex avec vue partielle sur le lac et les montagnes. Cet appartement se compose d...",Appartement,7.5,171 m2,R.D.C.,CHF 3'290.�,Immostreet
292,17/03/2017,Rent,Lausanne,Chemin de la Grangette 81C,1010 Lausanne,"Magnifique appartement lumineux situ� � deux pas du quartier de Chailly. Il est compos� de trois chambres (entre 12 et 14m2 chacune) dont une parentale avec salle de douche/wc, une salle de bains/w...",Appartement,4,97 m2,2.,CHF 3'400.�,Immostreet
293,17/03/2017,Rent,Lausanne,Chemin de la Cigale 2 A,1010 Lausanne,"www.courtillet.ch ""Le Parc du Courtillet"", mise en location de 21 logements neufs. Plusieurs appartements de 5,5 pi�ces surfaces d�s 122.30m2 avec une belle cuisine �quip�e, une salle de bain / WC ...",Appartement,5.5,122 m2,R.D.C.,CHF 3'410.�,Immostreet
294,17/03/2017,Rent,Lausanne,Chemin de la Cigale 2 C,1010 Lausanne,"www.courtillet.ch ""Le Parc du Courtillet"", mise en location de 21 logements neufs. Plusieurs appartements de 5,5 pi�ces surfaces d�s 122.30m2 avec une belle cuisine �quip�e, une salle de bain / WC ...",Appartement,5.5,122 m2,1.,CHF 3'410.�,Immostreet
295,17/03/2017,Rent,Lausanne,Chemin de Boissonnet 13A,1010 Lausanne,"Situ� dans une nouvelle construction de 2 petits immeubles contemporains de 3 appartements chacun, la r�sidence ""LES TERRASSES DE BOISSONNET"" offre une r�alisation de haut standing de type Minergie...",Appartement,4.5,143 m2,R.D.C.,CHF 3'980.�,Immostreet
296,17/03/2017,Rent,Lausanne,Ch. de Verdonnet 5,1010 Lausanne,"Dans quartier r�sidentiel, � 3 minutes � pied du CHUV et 12 de Saint-Fran�ois, Bus TL � la porte, proche de tout, magnifique logement avec jardin exceptionnel de 500 m2 cl�tur�, arrosage automatiqu...",Appartement,4.5,120 m2,R.D.C.,CHF 4'600.�,Immostreet
297,17/03/2017,Rent,Lausanne,Situation exeptionnelle,1010 Lausanne,"La propri�t� est situ�e aux portes de Lausanne, dans un endroit calme et verdoyant. Elle d�tient un grand parc richement arboris� avec des arbres s�culaires. L'appartement de 10 pi�ces se trouve da...",Maison individuelle,10,250 m2,,CHF 6'500.�,Immostreet
298,17/03/2017,Rent,Lausanne,Avenue de Valmont 16,1010 Lausanne,Magnifique appartement de 6 pi�ces en duplex au 13�me �tage comprenant : - hall d'entr�e - 4 chambres - s�jour - salle � manger - cuisine enti�rement agenc�e avec lave-vaisselle - salle de bains - ...,Appartement,6,150 m2,13.,,Immostreet
299,17/03/2017,Rent,Lausanne,Route de Berne 24,1010 Lausanne,"Immeuble id�alement situ� proche du CHUV, du M2 et de toutes les commodit�s. Petit d�p�t pour ranger du mat�riel",Cave,,,,CHF 80.�,Immostreet
300,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement d'une pi�ce, cuisine agenc�e ouverte sur le s�jour, salle de bains avec douche/W.-C. balcon. Transports publics, commerces et �coles enfa...",Appartement,1,27 m2,4.,CHF 960.�,Immostreet
301,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement d'une pi�ce, cuisine agenc�e ouverte sur le s�jour, salle de bains avec douche/W.-C. balcon. Transports publics, commerces et �coles enfa...",Appartement,1,31 m2,1.,CHF 1'020.�,Immostreet
302,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement d'une pi�ce, cuisine agenc�e ouverte sur le s�jour, salle de bains avec douche/W.-C. balcon. Transports publics, commerces et �coles enfa...",Appartement,1,31 m2,2.,CHF 1'050.�,Immostreet
303,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement d'une pi�ce, cuisine agenc�e ouverte sur le s�jour, salle de bains avec douche/W.-C. balcon. Transports publics, commerces et �coles enfa...",Appartement,1,31 m2,3.,CHF 1'090.�,Immostreet
304,17/03/2017,Rent,Lausanne,Chemin de la Fauvette2,1012 Lausanne,"Appartement de 1 pi�ce de ~ 39m2 au 2�me �tage, compos� comme suit: Hall, cuisine agenc�e, salle de bain avec WC, une chambre. L'appartement sera compl�tement r�nov�. Pour visiter le bien, merci de...",Appartement,1,39 m2,2.,CHF 1'220.�,Immostreet
305,17/03/2017,Rent,Lausanne,Avenue de Beaumont 26Bis,1012 Lausanne,"Excellente situation, � proximit� du CHUV et des transports publics. comprenant hall, s�jour, une chambre. Cuisine avec frigo, sans cuisini�re. Salle de bains/WC.",Appartement,2,45 m2,1.,CHF 1'270.�,Immostreet
306,17/03/2017,Rent,Lausanne,Ch. du Devin 57,1012 Lausanne,"Hall d'entr�e, cuisine ouverte avec frigo et cuisini�re, salle de bains-WC. 2 chambres. Pas de balcon. A louer pour le 16.04.2017. L'appartement sera repeint enti�rement. SEULES ET UNIQUES VISITES ...",Appartement,2.5,46 m2,2.,CHF 1'340.�,Immostreet
307,17/03/2017,Rent,Lausanne,Avenue de Chailly 16,1012 Lausanne,"A louer au 16 avril 2017, appartement de 2.5 pi�ces au 2e �tage, comprenant, entr�e, hall meublable, cuisine agenc�e, une salle de bains/WC, 1 chambre � coucher, salon avec balcon. Loyer mensuel : ...",Appartement,2,54 m2,2.,CHF 1'425.�,Immostreet
308,17/03/2017,Rent,Lausanne,Avenue de la Vallonnette 22,1012 Lausanne,"Sympathique appartement de 4 pi�ces au 1er �tage comprenant : - Hall, - Cuisine, - S�jour, - 3 chambres � coucher, - Salle-de-bains, - WC s�par�, - Balcon, - Cave, R�fection compl�te des peintures ...",Appartement,4,76 m2,1.,CHF 1'585.�,Immostreet
309,17/03/2017,Rent,Lausanne,Chemin de Champ-Soleil 1,1012 Lausanne,"Tr�s belle situation, au calme, dans le quartier de Chailly avec toutes les commodit�s � proximit� imm�diate. Appartement comprenant : hall d'entr�e, grand s�jour et salle � manger, une chambre � c...",Appartement,3,67 m2,2.,CHF 1'710.�,Immostreet
310,17/03/2017,Rent,Lausanne,Avenue de Chailly 16,1012 Lausanne,"A louer au 16 avril 2017, appartement de 3,5 pi�ces au 4e �tage, comprenant, entr�e, hall meublable, cuisine agenc�e avec frigo, une salle de bains/WC, 2 chambres � coucher, salon avec balcon (d�ga...",Appartement,3,74 m2,4.,CHF 1'750.�,Immostreet
311,17/03/2017,Rent,Lausanne,Diablerets 5,1012 Lausanne,"Appartement de 3 pi�ces au 1er �tage compos� d'un hall d'entr�e, un s�jour, une cuisine agenc�e, deux chambres � coucher, une salle de bains/WC, un WC s�par� et un balcon. Pour les visites, merci d...",Appartement,3,68 m2,1.,CHF 1'750.�,Immostreet
312,17/03/2017,Rent,Lausanne,Avenue de la Vallonnette 22,1012 Lausanne,"Joli appartement de 4.5 pi�ces au 3�me �tage comprenant : - Hall, - Cuisine, - S�jour, - 3 chambres � coucher, - Salle-de-bains, - WC s�par�, - Balcon, - Cave, Visites organis�e par le locataire ac...",Appartement,4.5,,3.,CHF 1'778.�,Immostreet
313,17/03/2017,Rent,Lausanne,Avenue Victor-Ruffy 19,1012 Lausanne,"Appartement de 3 pi�ces au 1er �tage dont les peintures ont �t� refaites, comprenant : hall, s�jour, 2 chambres, cuisine avec cuisini�re �lectrique, four, hotte, r�frig�rateur, salle de bains-wc, b...",Appartement,3,,1.,CHF 1'800.�,Immostreet
314,17/03/2017,Rent,Lausanne,Avenue du Temple 15,1012 Lausanne,"Quartier Chailly � Lausanne Appartement de 3 pi�ces au 4�me �tage, d'env. 75m2 Ce logement se compose de la mani�re suivante: hall d'entr�e, cuisine agenc�e, 2 chambres, s�jour, balcon, salle de ba...",Appartement,3,75 m2,4.,CHF 1'830.�,Immostreet
315,17/03/2017,Rent,Lausanne,chailly 56,1012 Lausanne,"A louer d�s le 20 mars 2017, appartement de 3,5 pces situ� � Chailly sur Lausanne. S�jour, cuisine ouverte et enti�rement �quip�e avec lave-vaisselle, salle de bains avec douche et wc, 2 chambres, ...",Appartement,3.5,60 m2,3.,CHF 1'950.�,Immostreet
316,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement de deux chambres, cuisine agenc�e ouverte sur le s�jour, salle de bains/W.-C., WC s�par�s, loggia. Transports publics, commerces et �cole...",Appartement,3.5,81 m2,2.,CHF 2'020.�,Immostreet
317,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement d'une chambre, cuisine agenc�e ouverte sur le s�jour, salle de bains/W.-C., grand balcon. Transports publics, commerces et �coles enfanti...",Appartement,2,52 m2,7.,CHF 2'050.�,Immostreet
318,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement de deux chambres, cuisine agenc�e ouverte sur le s�jour, salle de bains/W.-C., WC s�par�s, loggia. Transports publics, commerces et �cole...",Appartement,3.5,81 m2,3.,CHF 2'080.�,Immostreet
319,17/03/2017,Rent,Lausanne,,1012 Lausanne,"A louer splendide appartement �de 3.5 pi�ces refait � neuf,�situ� sous les combles avec beaucoupde charme et poutres apparentes. Il se r�partit de la mani�re suivante -�une cuisine agenc�e (4 plaqu...",App. mansard�,3,64 m2,4.,CHF 2'080.�,Immostreet
320,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement de deux chambres, cuisine agenc�e ouverte sur le s�jour, salle de bains/W.-C., WC s�par�s, loggia. Transports publics, commerces et �cole...",Appartement,3.5,81 m2,4.,CHF 2'120.�,Immostreet
321,17/03/2017,Rent,Lausanne,,1012 Lausanne,"Magnifique appartement de 3 pi�ces au centre de Chailly, compos� de : S�jour, cuisine agenc�e, salle de bain avec baignoire, deux chambres et une grande terrasse. L'appartement sera enti�rement r�n...",Appartement,3,72 m2,R.D.C.,CHF 2'140.�,Immostreet
322,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement de deux chambres, dressing, cuisine agenc�e ouverte sur le s�jour, salle de bains/W.-C., douche-WC, balcon. Transports publics, commerces...",Appartement,3.5,88 m2,4.,CHF 2'320.�,Immostreet
323,17/03/2017,Rent,Lausanne,Chemin de la Rosi�re 26,1012 Lausanne,"Dans un quartier calme et verdoyant, � louer tr�s joli appartement de 3,5 pi�ces avec magnifique vue sur le lac. Cet objet se compose comme suit : - hall d'entr�e - cuisine ferm�e enti�rement agenc...",Appartement,3.5,,,CHF 2'380.�,Immostreet
324,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement de deux chambres, cuisine agenc�e ouverte sur le s�jour, salle de bains/W.-C., WC s�par�s, loggia. Transports publics, commerces et �cole...",Appartement,3.5,81 m2,5.,CHF 2'420.�,Immostreet
325,17/03/2017,Rent,Lausanne,Chemin Beau-Val 3,1012 Lausanne,� Surface approx. 95 m2 � R�novation compl�te en 2011 avec finitions de standing � 3 pi�ces et demi comprenant : - 1 grand salon avec une chemin�e -1 cuisine ouverte de standing tout �quip� -2 cham...,Appartement,3.5,95 m2,1.,CHF 2'490.�,Immostreet
326,17/03/2017,Rent,Lausanne,Avenue de Chailly 63B,1012 Lausanne,"Immeuble situ� dans le quartier de Chailly � Lausanne � proximit� des transports publics, commerces et �coles. Magnifique appartement de 3.5 pi�ces au rez sup�rieur compos� d'un hall d'entr�e, une ...",Appartement,3.5,84 m2,R.D.C.,CHF 2'550.�,Immostreet
327,17/03/2017,Rent,Lausanne,Ch. du Devin 96,1012 Lausanne,"Nous vous proposons un b�timent d'un standing minergie, r�nov� en 2014. Spacieux logement avec cuisine enti�rement �quip�e ouverte sur un grand s�jour, 2 chambres, une salle de bains avec emplaceme...",Appartement,3.5,111 m2,1.,CHF 2'590.�,Immostreet
328,17/03/2017,Rent,Lausanne,Av. de Chailly 52,1012 Lausanne,"Bel appartement de 5 pi�ces au 4e �tage, ce dernier comprend : un grand hall, corridor, grand salon, salle � manger, une cuisine �quip�e, un coin � manger, une salle de bains, WC s�par�, trois cham...",Appartement,5.5,109 m2,6.,CHF 2'725.�,Immostreet
329,17/03/2017,Rent,Lausanne,Avenue du Temple 12,1012 Lausanne,"Quartier Chailly Bel appartement en duplex de 4.5 pi�ces, d'environ 105m2, situ� au 2�me �tage d'une propri�t� entour�e d'un superbe �crin de verdure Situ� en hauteur, l�g�rement en retrait des nui...",Appartement,4.5,100 m2,2.,CHF 2'740.�,Immostreet
330,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement de trois chambres, dressing, cuisine agenc�e ouverte sur le s�jour, salle de bains/W.-C., douche-WC, balcon. Transports publics, commerce...",Appartement,4,101 m2,2.,CHF 2'980.�,Immostreet
331,17/03/2017,Rent,Lausanne,Sur demande,1012 Lausanne,"Lumineux appartement avec terrasse, deck et jardin dans un environnement bucolique de verdure et de tranquillit� unique. Il offre 2 chambres � coucher, une cuisine enti�rement am�nag�e ouverte sur ...",Appartement,3.5,80 m2,R.D.C.,CHF 3'000.�,Immostreet
332,17/03/2017,Rent,Lausanne,Ch. de Haute-Vue 1a,1012 Lausanne,Lumineux appartement avec terrasse deck et jardin dans un environnement bucolique de verdure et de tranquillit� unique. Appartement neuf richement �quip� et confortable avec 2 salles d'eau dont l'u...,Appartement,3.5,80 m2,R.D.C.,CHF 3'050.�,Immostreet
333,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement de trois chambres, dressing, cuisine agenc�e ouverte sur le s�jour, salle de bains/W.-C., douche-WC, balcon. Transports publics, commerce...",Appartement,4,101 m2,3.,CHF 3'060.�,Immostreet
334,17/03/2017,Rent,Lausanne,Chemin de Haute-Brise 15,1012 Lausanne,"Villa locative comprenant 3 appartements, situ�e dans les hauts de Chailly. Quartier tranquille et verdoyant. Comprenant: hall d'entr�e, 4 chambres � coucher, s�jour avec chemin�e, salle � manger, ...",Appartement,6,120 m2,1.,CHF 3'080.�,Immostreet
335,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement de trois chambres, dressing, cuisine agenc�e ouverte sur le s�jour, salle de bains/W.-C., douche-WC, balcon. Transports publics, commerce...",Appartement,4.5,103 m2,5.,CHF 3'190.�,Immostreet
336,17/03/2017,Rent,Lausanne,Chemin de la Fauvette 69,1012 Lausanne,"Situ�e dans un environnement r�sidentiel � seulement 3 minutes du charmant quartier de Chailly et de ses nombreux commerces, la maison est bien entretenue, lumineuse et se compose comme suit : Rez-...",Villa,4.5,120 m2,,CHF 3'200.�,Immostreet
337,17/03/2017,Rent,Lausanne,Ch. des Bouleaux 12,1012 Lausanne,Quartier Chailly. 1�re location dans immeuble neuf. Appartement �quip� de mat�riaux de qualit�. Salon avec cuisine ouverte acc�s balcon-terrasse. Chambre parentale avec dressing et salle de bains p...,Appartement,4.5,104 m2,3.,CHF 3'370.�,Immostreet
338,17/03/2017,Rent,Lausanne,Chemin de Rov�r�az 28A,1012 Lausanne,"Ce bien d�exception se situe � deux pas du centre-ville, des transports publics, des �coles (�cole Nouvelle de la Suisse Romande), des cr�ches et des commerces. Les acc�s autoroutiers sont � 5-10 m...",Duplex,4.5,,1.,CHF 3'450.�,Immostreet
339,17/03/2017,Rent,Lausanne,Ch. des Bouleaux 12,1012 Lausanne,Quartier Chailly. 1�re location dans immeuble neuf. Appartement �quip� de mat�riaux de qualit�. Salon avec cuisine ouverte acc�s balcon-terrasse. Chambre parentale avec dressing et salle de bains p...,Appartement,4.5,104 m2,4.,CHF 3'570.�,Immostreet
340,17/03/2017,Rent,Lausanne,Avenue de Chailly 12,1012 Lausanne,"A louer dans immeuble neuf de standing Spacieux appartement de trois chambres, cuisine agenc�e ouverte sur le s�jour, salle de bains/W.-C., douche-WC, r�duit, grand balcon. Transports publics, comm...",Appartement,4.5,129 m2,6.,CHF 3'770.�,Immostreet
341,17/03/2017,Rent,Lausanne,Chemin de la Cure 9 b,1012 Lausanne,"Appartement de 4,5 pi�ces au 1er �tage d'environ 117 m2 dans une villa locative neuve sur les hauts de Chailly et comprenant un hall d'entr�e, une cuisine enti�rement agenc�e (four + four vapeur, l...",Appartement,4.5,112 m2,1.,CHF 3'780.�,Immostreet
342,17/03/2017,Rent,Lausanne,Rue de la Fauvette 57,1012 Lausanne,"A louer au 1er juillet 2017 ou date � convenir. Villa de 5.5 pi�ces avec jardin � la rue de la Fauvette 57 � Lausanne. Situ�e a quelques minutes du centre-ville de Lausanne, ce bien profite d'une s...",Villa,5.5,,,CHF 3'900.�,Immostreet
343,17/03/2017,Rent,Lausanne,Grand-Praz 5,1012 Lausanne,"A louer superbe appartement de 5 pi�ces dans un immeuble de standing en PPE, proche de la ligne No 7 et de toutes commodit�s tout en �tant dans un quartier calme. Il comprend : un hall, un spacieux...",Appartement,5,,,CHF 4'000.�,Immostreet
344,17/03/2017,Rent,Lausanne,chemin de Haute-brise 7C,1012 Lausanne,"Profitant d'un cadre de vie id�al, situ� dans les beaux quartiers de Chailly-sur-Lausanne, proche de toutes commodit�s et commerces, ce bel appartement b�n�ficie de belles finitions et d'un mobilie...",Appartement,5.5,150 m2,R.D.C.,CHF 4'400.�,Immostreet
345,17/03/2017,Rent,Lausanne,Sur demande,1012 Lausanne,"Superbe villa individuelle sur 3�tages proche du centre-ville de Lausanne et de toutes commodit�s (5min � pied des commerces). Situ�e dans un quartier r�sidentiel calme et verdoyant, � l'abri des n...",Maison individuelle,6.5,200 m2,,CHF 5'200.�,Immostreet
346,17/03/2017,Rent,Lausanne,"2, chemin de Craivavers",1012 Lausanne,"Situ�e dans un quartier r�sidentiel calme et verdoyant de Chailly, � 5 min du centre ville de Lausanne et des axes autoroutiers, cette charmante maison dispose d'une surface de 230 m2. Cette maison...",Villa,7.5,230 m2,,CHF 6'900.�,Immostreet
347,17/03/2017,Rent,Lausanne,Chemin de Craivavers 2,1012 Lausanne,"Villa de haut standing 230m2, enti�rement r�nov�e en 2016 avec go�t. Tr�s lumineuse, cette belle maison de plein pied est entour�e d'un magnifique jardin richement arbor� sur une parcelle de 1800m2...",Maison individuelle,7.5,230 m2,,CHF 6'900.�,Immostreet
348,17/03/2017,Rent,Lausanne,Chemin de la Fauvette 31A,1012 Lausanne,"Situ� dans un immeuble r�cent, compos� de seulement 4 appartements, ce sublime appartement meubl� b�n�ficie d'une situation privil�gi�e dans un quartier calme et r�sidentiel tout en �tant � proximi...",Appartement,5.5,150 m2,R.D.C.,CHF 7'150.�,Immostreet
349,17/03/2017,Rent,Lausanne,Chemin de la Cure 19,1012 Lausanne,"Superbe spacieuse propri�t� d'env. 300m2 enti�rement r�nov�e, situ�e dans un quartier r�sidentiel pris� et � deux pas du plateau de Chailly. Rez-de-chauss�e Hall d'entr�e avec penderie, cage d'esca...",Villa,9,300 m2,,CHF 11'000.�,Immostreet
350,17/03/2017,Rent,Lausanne,Chemin d' Entrebois 17,1018 Lausanne,"Appartement subventionn� de 1 pi�ce au quatri�me �tage, cuisine, pi�ce principale, salle de bains-WC. Balcon. Visites du logement: le 16.03.17 de 17h30-18h30 et le 18.03.2017 de 11h00 � 12h00.",Appartement,1,27 m2,4.,CHF 253.�,Immostreet
351,17/03/2017,Rent,Lausanne,Rte du Signal 17,1018 Lausanne,"Dans immeuble de caract�re, situ� dans un parc, jolie chambre ind�pendante pour �tudiant avec armoire, haut plafond, parquet massif au sol, lavabo. WC/douche sur le palier. Immeuble raccord� � la f...",Chambre,1,,1.,CHF 450.�,Immostreet
352,17/03/2017,Rent,Lausanne,Chemin d' Entrebois 17,1018 Lausanne,"comprenant: hall, s�jour, deux chambres � coucher, cuisine, salle de bains-WC. Appartement subventionn�. Pour les visites: tous les lundis de 18h00 � 19h00.",Appartement,3,63 m2,13.,CHF 504.�,Immostreet
353,17/03/2017,Rent,Lausanne,Rte du Signal 17,1018 Lausanne,"Dans immeuble de caract�re, situ� dans un parc, chambre ind�pendante pour �tudiant comprenant entr�e avec armoire , coin lavabo et balcon. Id�al pour �tudiant/e (WC/douche sur le palier). Bus no 16...",Chambre,1,,1.,CHF 530.�,Immostreet
354,17/03/2017,Rent,Lausanne,Route Aloys-Fauquez 147,1018 Lausanne,"situ� dans un quartier calme, proche des transports et commerces, hall, cuisine, salle de bains-WC, balcon.",Appartement,1,,3.,CHF 755.�,Immostreet
355,17/03/2017,Rent,Lausanne,Ch. Des Sauges,1018 Lausanne,"Appartement de 1pi�ce (30 m2) au rez sup�rieur. Cuisinette, s�jour avec balcon, salle de bains-WC. Situation calme � proximit� de toutes commodit�s, quartier Bl�cherette.",Appartement,1,30 m2,R.D.C.,CHF 930.�,Immostreet
356,17/03/2017,Rent,Lausanne,Chemin du Furet 10,1018 Lausanne,"Appartement pratique et lumineux, comprenant: une cuisine agenc�e, salle de bains et une pi�ce � vivre avec parquet. La peinture et le parquet de l'appartement seront refait. L' immeuble dispose d'...",Appartement,1,28 m2,1.,CHF 1'100.�,Immostreet
357,17/03/2017,Rent,Lausanne,Ch. de Maillefer 23,1018 Lausanne,"Situ�s dans un nouvel immeuble de 12 logements et proche de toutes commodit�s, ces 3 studios, se composent comme suit: - Pi�ce � vivre avec acc�s balcon - Cuisine ouverte enti�rement agenc�e - Sall...",Appartement,1,23 m2,1.,CHF 1'130.�,Immostreet
358,17/03/2017,Rent,Lausanne,Borde 15,1018 Lausanne,"Appartement de 2.5 pi�ces au 4�me �tage compos� d'un hall d'entr�e, un s�jour, une cuisine agenc�e, une chambre � coucher, une salle de bains/WC, un balcon. Pour les visites, merci de contacter le ...",Appartement,2.5,52 m2,4.,CHF 1'200.�,Immostreet
359,17/03/2017,Rent,Lausanne,CH. DU FURET 6,1018 Lausanne,"Cuisine : fen�tre, habitable ferm�e / Agencement : frigo / Caract�ristiques : balcon, cave . Orientation : OUEST",Appartement,2,50 m2,2.,CHF 1'300.�,Immostreet
360,17/03/2017,Rent,Lausanne,Chemin de M�mise 8,1018 Lausanne,"Joli appartement de 2 pi�ces au rez, comprenant : Chambre � coucher, salle de bains, cuisine agenc�e avec frigo et cuisini�re, salon. L'immeuble est dot� d'un interphone et d'un ascenseur. Disponib...",Appartement,2,41 m2,R.D.C.,CHF 1'315.�,Immostreet
361,17/03/2017,Rent,Lausanne,Avenue du Grey 43,1018 Lausanne,"Entr�e, cuisine agenc�e avec frigo, cuisini�re avec four et plan de cuisson vitroc�ram, hotte de ventilation et lave-vaisselle, salon, une chambre � coucher, salle de bains/WC, un balcon.",Appartement,2,42 m2,R.D.C.,CHF 1'375.�,Immostreet
362,17/03/2017,Rent,Lausanne,Ch. Des Sauges,1018 Lausanne,"Appartement de 2 pi�ces (51 m2) au 1er �tage. Hall d'entr�e avec armoires murales, cuisine agenc�e, salle de bains-WC, s�jour avec balcon, 1 chambre � coucher. Quartier Bl�cherette. Situation calme...",Appartement,2,51 m2,1.,CHF 1'410.�,Immostreet
363,17/03/2017,Rent,Lausanne,Rte du Pavement 87,1018 Lausanne,"Appartement enti�rement r�nov�. Cuisine agenc�e, salle de bains-WC.",Appartement,3,62 m2,1.,CHF 1'415.�,Immostreet
364,17/03/2017,Rent,Lausanne,Rte du Ch�telard 4,1018 Lausanne,"Appartement enti�rement r�nov�. Cuisine agenc�e y.c. lave-vaisselle, salle de bains-WC, balcon. R�serv� : un contrat est en cours pour cet objet.",Appartement,3.5,67 m2,2.,CHF 1'530.�,Immostreet
365,17/03/2017,Rent,Lausanne,Rue de la Pontaise 40,1018 Lausanne,"Appartement pratique et lumineux, comprenant: hall, salon avec coin � manger et balcon, chambre-�-coucher, cuisine semi-ouverte, salle-de-bains avec baignoire. L' immeuble dispose d'un ascenseur. S...",Appartement,2,54 m2,4.,CHF 1'570.�,Immostreet
366,17/03/2017,Rent,Lausanne,RTE DU PAVEMENT 15,1018 Lausanne,"Cuisine : fen�tre, habitable ferm�e / Agencement : frigo / Caract�ristiques : balcon, hall, cave . Orientation : OUEST, SUD",Appartement,3,65 m2,5.,CHF 1'630.�,Immostreet
367,17/03/2017,Rent,Lausanne,Ch. de Maillefer 23,1018 Lausanne,"Situ� dans un nouvel immeuble de 12 logements et proche de toutes commodit�s, cet appartement de 2.5 pi�ces au rez-de-chauss�e, se compose comme suit: - Hall d'entr�e - S�jour avec acc�s � la terra...",Appartement,2.5,42 m2,R.D.C.,CHF 1'655.�,Immostreet
368,17/03/2017,Rent,Lausanne,CH. DU PETIT-FLON 56,1018 Lausanne,"Cuisine : laboratoire, ouverte s/coin manger / Agencement : frigo, cuisini�re � gaz / Caract�ristiques : hall, pelouse, cave . Orientation : EST",Appartement,3.5,71 m2,R.D.C.,CHF 1'710.�,Immostreet
369,17/03/2017,Rent,Lausanne,Ch. de Maillefer 23,1018 Lausanne,"Situ�s dans un nouvel immeuble de 12 logements et proche de toutes commodit�s, ces 3 appartements de 2.5 pi�ces, se composent comme suit: - Hall d'entr�e - S�jour avec acc�s balcon - Cuisine ouvert...",Appartement,2.5,47 m2,1.,CHF 1'760.�,Immostreet
370,17/03/2017,Rent,Lausanne,Ch. de Maillefer 23,1018 Lausanne,"Situ� dans un nouvel immeuble de 12 logements et proche de toutes commodit�s, cet appartement de 3.5 pi�ces au rez-de-chauss�e, se compose comme suit: - Hall d'entr�e avec armoires murales - S�jour...",Appartement,3.5,58 m2,R.D.C.,CHF 1'925.�,Immostreet
371,17/03/2017,Rent,Lausanne,Rue de la Pontaise 52,1018 Lausanne,Ce logement partiellement r�nov� est dispos� comme suit : hall d'entr�e avec armoire murale s�jour lumineux avec balcon cuisine agenc�e et ouverte sur le s�jour 2 chambres salle-de-bains/WC. Dispon...,Appartement,3.5,63 m2,1.,CHF 1'980.�,Immostreet
372,17/03/2017,Rent,Lausanne,Route du Pavement 75 A,1018 Lausanne,"Dans un quartier agr�able, proche de toutes les commodit�s Appartement de 2.5 pi�ces au 4�me �tage, comprenant : une chambre, une cuisine ouverte sur le s�jour, une salle de bains/wc, un balcon et ...",Appartement,2.5,75 m2,4.,CHF 2'000.�,Immostreet
373,17/03/2017,Rent,Lausanne,,1018 Lausanne,"Magnifique appartement de 3,5 pi�ces qui se trouve au chemin de la For�t 4 B, 1018 Lausanne. Il est compos� d'une pi�ce � vivre avec cuisine ouverte enti�rement �quip�e, de 2 chambres � coucher don...",Appartement,3,,,CHF 2'200.�,Immostreet
374,17/03/2017,Rent,Lausanne,Ch. de Maillefer 23,1018 Lausanne,"Situ� dans un nouvel immeuble de 12 logements et proche de toutes commodit�s, cet appartement de 3.5 pi�ces, se compose comme suit: - Hall d'entr�e avec armoires murales - Grand s�jour avec acc�s b...",Appartement,3.5,75 m2,1.,CHF 2'295.�,Immostreet
375,17/03/2017,Rent,Lausanne,Ch. de Maillefer 23,1018 Lausanne,"Situ�s dans un nouvel immeuble de 12 logements et proche de toutes commodit�s, ces deux appartements de 3.5 pi�ces, se composent comme suit: - Hall d'entr�e avec armoires murales - Grand s�jour ave...",Appartement,3.5,75 m2,2.,CHF 2'375.�,Immostreet
376,17/03/2017,Rent,Lausanne,Pavement 105,1018 Lausanne,"Dans un petit immeuble, magnifique appartement de 3.5 pi�ces comprenant entr�e, deux chambres � coucher, salon avec chemin�e, cuisine agenc�e, SB/douche, WC s�par�s et grand balcon donnant sur le j...",Appartement,3.5,,1.,CHF 2'500.�,Immostreet
377,17/03/2017,Rent,Lausanne,Av. de Gratta-Paille 12,1018 Lausanne,"Cet appartement se distingue par sa situation privil�gi�e, ainsi que par ses pi�ces innond�es de lumi�re avec les atouts suivants : - Cuisine agenc�e (y.c. lave-vaisselle) - Grand salon avec du par...",Appartement,4.5,97 m2,1.,CHF 2'530.�,Immostreet
378,17/03/2017,Rent,Lausanne,Ch. des Sauges 18,1018 Lausanne,"Superbe appartement de 4 � pi�ces n� 103 d'environ 96 m2 au 1er �tage comprenant : hall, s�jour, cuisine ouverte sur salon et agenc�e avec lave-vaisselle, salle de bains-WC, douche-WC, loggia. Buan...",Appartement,4.5,96 m2,1.,CHF 2'570.�,Immostreet
379,17/03/2017,Rent,Lausanne,Route des Plaines-du-Loup 64,1018 Lausanne,"A louer de suite ou � convenir, ce bel appartement neuf en PPE situ� � seulement quelques arr�ts de bus du centre ville. Il offre une surface habitable de 114 m2, dont un s�jour/salle � manger avec...",Appartement,4.5,114 m2,1.,CHF 2'600.�,Immostreet
380,17/03/2017,Rent,Lausanne,Avenue Parc-de-la-Rouvraie 25,1018 Lausanne,"Ce logement spacieux et lumineux, saura vous charmer gr�ce � la qualit� des ses am�nagements � savoir : - Cuisine agenc�e ferm�e (y compris lave-vaisselle) - S�jour avec du parquet - 3 Chambres � c...",Appartement,4.5,101 m2,4.,CHF 2'660.�,Immostreet
381,17/03/2017,Rent,Lausanne,Av. de Gratta-Paille 12,1018 Lausanne,"Cet appartement se distingue par sa situation particuli�re, ainsi que par ses pi�ces innond�es de lumi�re avec les atouts suivants : - Cuisine agenc�e (y.c. lave-vaisselle) - Grand salon avec du pa...",Appartement,4.5,111 m2,1.,CHF 2'670.�,Immostreet
382,17/03/2017,Rent,Lausanne,Sur demande,1018 Lausanne,"Superbe appartement de 4.5 pi�ces 100 m�tres carr� � Lausanne, Il dispose de trois chambres, deux salles de bain, un s�jour, un hall, une cuisine �quip�, un jardin avec terrasse, une cave et une pl...",Appartement,4.5,130 m2,R.D.C.,CHF 2'700.�,Immostreet
383,17/03/2017,Rent,Lausanne,Avenue Parc-de-la-Rouvraie 12,1018 Lausanne,"Nous vous rendons attentifs que la cuisine ainsi que les sanitaires seront modernis�s durant l'ann�e 2017. Ce logement se situe dans un quartier calme et verdoyant, id�al pour une famille avec les ...",Appartement,5,120 m2,7.,CHF 2'710.�,Immostreet
384,17/03/2017,Rent,Lausanne,Parc-de-La-Rouvraie 26A,1018 Lausanne,"Parc-de-La-Rouvraie 26A � Lausanne. Dans un quartier calme et verdoyant, � louer magnifique appartement au 1er �tage avec sur le lac. Cet objet se compose comme suit: - hall d'entr�e - cuisine ferm...",Appartement,4.5,110 m2,1.,CHF 2'780.�,Immostreet
385,17/03/2017,Rent,Lausanne,,1018 Lausanne,"Tr�s beau loft situ� dans le quartier de Bellevaux, Chemin de la for�t 4A Il est compos� d'une grande pi�ce � vivre avec cuisine ouverte et enti�rement agenc�e, 3 chambres � coucher avec 2 salles d...",Loft,4,,R.D.C.,CHF 2'800.�,Immostreet
386,17/03/2017,Rent,Lausanne,Quartier de Violette 6,1018 Lausanne,"Situ�e dans un quartier r�sidentiel, tr�s calme au centre-ville, cette r�alisation de standing offre une situation tr�s agr�able et pris�e. L'immeuble est id�alement plac�, proche de toutes les com...",Appartement,3.5,101 m2,1.,CHF 2'900.�,Immostreet
387,17/03/2017,Rent,Lausanne,Route du Signal 17 bis,1018 Lausanne,"Dans le calme et la verdure d'un magnifique parc arboris� (id�al pour les enfants), � 2 pas du Bois de Sauvabelin, spacieux appartement traversant enti�rement r�nov� de 5 pi�ces et hall au 4�me �ta...",Appartement,5.5,110 m2,4.,CHF 2'950.�,Immostreet
388,17/03/2017,Rent,Lausanne,Route du Signal 17 bis,1018 Lausanne,"Dans le calme et la verdure d'un parc � 2 pas du Bois de Sauvabelin spacieux appartement traversant de 6 pi�ces et hall au 2�me �tage comprenant 4 chambres � coucher, grand salon/salle � manger ave...",Appartement,6.5,125 m2,2.,CHF 3'010.�,Immostreet
389,17/03/2017,Rent,Lausanne,Quartier de la Violette 6,1018 Lausanne,"DETAILS: Comfortable, modern 2-bedroom, 2 bathroom apartment available March 1st (or earlier) in a quiet neighborhood in Lausanne. Apartment is 100m2. 3.5 rooms; 1st floor. New building built appro...",Appartement,3.5,100 m2,1.,CHF 3'100.�,Immostreet
390,17/03/2017,Rent,Lausanne,Chemin des Grandes-Roches 9,1018 Lausanne,"Lease take over of 120m2, 4.5P, 1st floor unfurnished apartment. Available from 1st April 2017. 10 mins walk from Riponne, 5 mins from A9 Motorway. Comprising 3 double bedrooms. Main bedroom includ...",Appartement,4.5,100 m2,1.,CHF 3'300.�,Immostreet
391,17/03/2017,Rent,Lausanne,Av. Gratta-Paille 5,1018 Lausanne,Cet appartement se distingue par sa situation privil�gi�e proche de l'acc�s autoroutier ( Bl�cherette) avec les caract�ristiques suivantes : - Cuisine enti�rement agenc�e (y.c. lave-vaisselle) - Sa...,Appartement,5.5,140 m2,5.,CHF 3'510.�,Immostreet
392,17/03/2017,Rent,Lausanne,Avenue Parc-de-la-Rouvraie 12,1018 Lausanne,Nous vous rendons attentifs que la cuisine ainsi que les sanitaires seront modernis�s durant l'ann�e 2017. Ce logement en attique saura vous s�duire gr�ce aux donn�es suivantes : - Cuisine (sans la...,Appartement,7,174 m2,8.,CHF 4'165.�,Immostreet
393,17/03/2017,Rent,Lausanne,Ch. des Grandes-Roches 3a,1018 Lausanne,"1 spacieux hall d'entr�e avec armoires murales 1 cuisine agenc�e moderne, habitable et ouverte sur un coin � manger 1 grand s�jour lumineux avec vue imprenable sur le lac 1 chambre � coucher avec s...",Attique,5.5,156 m2,5.,CHF 5'340.�,Immostreet
394,17/03/2017,Rent,Lausanne,Route de Cojonnex 33 - 35,1000 Lausanne,Magnifique chambres meubl�es dans villa standing de 3 �tages avec grand jardin et places de parc en face de l'�cole EHL. Pour tout renseignement s'adresser � Mme GROGG au 079 449 21 79,Villa,6,,,CHF 960.�,Immostreet
395,17/03/2017,Rent,Lausanne,Ch. de la Vulliette 12,1000 Lausanne,Libre d�s le 1er juin 2017 Ce magnifique studio neuf se situe au ch. de la Vulliette 12 � Lausanne. Un arr�t de bus � c�t� de l'immeuble permet de rejoindre les Croisettes � Epalinges en quelques m...,Appartement,1,24 m2,R.D.C.,CHF 1'150.�,Immostreet
396,17/03/2017,Rent,Lausanne,,1000 Lausanne,"Entour� de verdure et dans un calme aboslu, cet appartement proche de toutes commodit�s est situ� � seulement quelques minutes de l'�cole h�teli�re, ainsi que du centre de Lausanne dans un immeuble...",Attique,3.5,80 m2,,CHF 1'790.�,Immostreet
397,17/03/2017,Rent,Lausanne,,1000 Lausanne,"Urgent! Nous quittons�l'appartement de 84m a fin de d�cembre il est bien situ� proche de commerces (denner, coop, migros, pharmacie, gym) �coles, garderies. 1 halle de entr� avec armoire murale 1 g...",Appartement,3.5,84 m2,3.,CHF 2'000.�,Immostreet
398,17/03/2017,Rent,Lausanne,Route du Jorat 190,1000 Lausanne,"Magnifique appartement prot�g� de 3.5 pi�ces de 80m2 � louer � la route du Jorat 190 � Pra Roman. Ce bel appartement prot�g�, d�di� prioritairement aux personnes s�niors (d�s 55 ans) ou � mobilit� ...",Appartement,3.5,80 m2,R.D.C.,CHF 2'075.�,Immostreet
399,17/03/2017,Rent,Lausanne,Route de Lo�x,1000 Lausanne,"Dans copropri�t� avec piscine et b�n�ficiant d'un joli parc, appartement de 3 pi�ces r�nov� r�cemment, d'environ 60 m2 au 2�me �tage comprenant : un hall, une salle de bains / wc avec cagibi, un s�...",Appartement,3,60 m2,,CHF 2'200.�,Immostreet
400,17/03/2017,Rent,Lausanne,Ch. des Molliettes 3,1000 Lausanne,"Dans une maison de charme datant de 1914 et totalement r�nov�e en 2008, superbe appartement de 3,5 pi�ces d'env. 90 m2, de haut standing avec un grand balcon. Un jardin est �galement � disposition ...",Appartement,3.5,,1.,CHF 2'459.�,Immostreet
401,17/03/2017,Rent,Lausanne,Chemin du Chalet-de-Praroman 4E,1000 Lausanne,"Immeuble neuf labelis� ""Minergie"" de 12 appartements sis dans un cadre de verdure et beau d�gagement sur les Alpes. Toutes commodit�s � proximit�. Comprenant : hall, cuisine agenc�e ouverte, grand ...",Appartement,4.5,111 m2,R.D.C.,CHF 2'740.�,Immostreet
402,17/03/2017,Rent,Lausanne,Chemin du Chalet-de-Praroman 4E,1000 Lausanne,"Immeuble neuf labelis� ""Minergie"" de 12 appartements sis dans un cadre de verdure et beau d�gagement sur les Alpes. Toutes commodit�s � proximit�. Comprenant : hall, cuisine agenc�e ouverte, grand ...",Appartement,3.5,104 m2,3.,CHF 2'750.�,Immostreet
403,17/03/2017,Rent,Lausanne,Chemin du Chalet-de-Praroman 4E,1000 Lausanne,"Immeuble neuf labelis� ""Minergie"" de 12 appartements sis dans un cadre de verdure et beau d�gagement sur les Alpes. Toutes commodit�s � proximit�. Comprenant : hall, cuisine agenc�e ouverte, grand ...",Appartement,3.5,116 m2,3.,CHF 2'965.�,Immostreet
404,17/03/2017,Rent,Lausanne,Chemin du Chalet-de-Praroman 4E,1000 Lausanne,"Immeuble neuf labelis� ""Minergie"" de 12 appartements sis dans un cadre de verdure et beau d�gagement sur les Alpes. Toutes commodit�s � proximit�. Comprenant : hall, cuisine agenc�e ouverte avec ba...",Appartement,4.5,99 m2,R.D.C.,CHF 3'040.�,Immostreet
405,17/03/2017,Rent,Lausanne,Ch. de la Planche-aux-Oies 15C,1000 Lausanne,"Aux portes de Lausanne, � 10 minutes � pied de l�Ecole H�teli�re de Lausanne et du Centre de recherche Nestl� de Vers-chez-les-Blancs, nous vous proposons ce tr�s bel objet qui vous offrira un cadr...",Maison  contigu�,5.5,178 m2,,CHF 3'490.�,Immostreet
406,17/03/2017,Rent,Lausanne,Route du Jorat 16,1000 Lausanne,"Dans un quartier calme, charmante villa individuelle de 5.5 pi�ces desservie sur trois niveaux.",Maison individuelle,5.5,200 m2,,CHF 3'990.�,Immostreet
407,17/03/2017,Rent,Lausanne,Residential,1000 Lausanne,"Beautiful modern house of 7.5 pi�ces around 285 m2 habitable located in a quiet, green and residential zone near Lausanne. This house is built on a field of 800m2 without vis-�-vis and a lot of sun...",Maison individuelle,7.5,,,CHF 5'000.�,Immostreet
408,17/03/2017,Rent,Lausanne,Residentiel,1000 Lausanne,"Tr�s belle maison contemporaine de 7.5 pi�ces jouissant de 285 m2 habitables (375 m2 utiles) situ�e dans un quartier r�sidentiel, calme et verdoyant pr�s de Lausanne. Cette villa est implant�e sur ...",Maison individuelle,7.5,,,CHF 5'000.�,Immostreet
409,17/03/2017,Rent,Lausanne,Chemin de la Planche-aux-Oies 6,1000 Lausanne,"Situation � 5 min. de Lausanne, � Vers-chez-les-Blancs, dans un environnement calme et verdoyant, acc�s par bus au m�tro Croisettes, � 18 min. en voiture de Vevey. Bus scolaire pour Ecole Internati...",Villa,9,250 m2,,CHF 6'000.�,Immostreet
410,17/03/2017,Rent,Lausanne,,1000 Lausanne,"Entour� de verdure et dans un calme aboslu, cet appartement proche de toutes commodit�s est situ� � seulement quelques minutes de l'�cole h�teli�re, ainsi que du centre de Lausanne dans un immeuble...",Appartement,4.5,116 m2,,,Immostreet
                                                                                                                                                                                                                                            Untitled.ipynb                                                                                      000644  000766  000024  00000010304 13137603537 015611  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Addition with variables: 5\n",
      "Multiplication with variables: 6\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "\n",
    "a = tf.placeholder(tf.int16)\n",
    "b = tf.placeholder(tf.int16)\n",
    "\n",
    "# Define some operations\n",
    "add = tf.add(a, b)\n",
    "mul = tf.multiply(a, b)\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    print (\"Addition with variables: %i\" % sess.run(add, feed_dict={a:2, b:3}))\n",
    "    print (\"Multiplication with variables: %d\" % sess.run(mul, feed_dict={a:2, b:3}))\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From <ipython-input-8-b9eb3c5fa28f>:25: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.\n",
      "Instructions for updating:\n",
      "Use `tf.global_variables_initializer` instead.\n",
      "0 0.164772 [ 1.23811221] [-0.11989033]\n",
      "100 3.91282e-05 [ 1.00726509] [-0.01651528]\n",
      "200 3.01224e-07 [ 1.00063753] [-0.00144907]\n",
      "300 2.31996e-09 [ 1.00005591] [-0.0001271]\n",
      "400 1.83806e-11 [ 1.00000501] [ -1.12432654e-05]\n",
      "500 1.71714e-13 [ 1.00000048] [ -1.00716125e-06]\n",
      "600 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "700 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "800 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "900 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "1000 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "1100 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "1200 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "1300 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "1400 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "1500 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "1600 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "1700 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "1800 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "1900 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n",
      "2000 3.78956e-14 [ 1.00000036] [ -6.25691371e-07]\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "\n",
    "x_data = [1,2,3]\n",
    "y_data = [1,2,3]\n",
    "\n",
    "# Try to find values for W and b taht compute y_data = W * x_data + b\n",
    "# (We know that W should be 1 and b 0, but Tensorflow will figure that out for us.)\n",
    "\n",
    "W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))\n",
    "b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))\n",
    "\n",
    "# Our hypothesis\n",
    "hypothesis = W * x_data + b\n",
    "\n",
    "# Simplified cost function\n",
    "cost = tf.reduce_mean(tf.square(hypothesis-y_data))\n",
    "\n",
    "# Minimize\n",
    "a = tf.Variable(0.1) # Learning rate, alpha\n",
    "optimizer = tf.train.GradientDescentOptimizer(a)\n",
    "train = optimizer.minimize(cost)\n",
    "\n",
    "# Before starting, initialize the variables.\n",
    "# We are going to run this first.\n",
    "init = tf.initialize_all_variables()\n",
    "\n",
    "# Launch the graph\n",
    "sess = tf.Session()\n",
    "sess.run(init)\n",
    "\n",
    "# Fit the line.\n",
    "for step in range(2001):\n",
    "    sess.run(train)\n",
    "    if step % 100 == 0 :\n",
    "        print (step, sess.run(cost), sess.run(W), sess.run(b))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
                                                                                                                                                                                                                                                                                                                            data-01-test-score.csv                                                                              000644  000766  000024  00000000101 13137603537 016702  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         93,88,93,185
89,91,90,180
96,98,100,196
73,66,70,142
53,46,55,101                                                                                                                                                                                                                                                                                                                                                                                                                                                               data-02-test-score.csv                                                                              000644  000766  000024  00000000101 13137603537 016703  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         93,88,93,183
89,90,90,179
93,97,100,194
73,66,71,143
53,46,54,100                                                                                                                                                                                                                                                                                                                                                                                                                                                               data-03-diabetes.csv                                                                                000644  000766  000024  00000150276 13137603537 016417  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         -0.294118,0.487437,0.180328,-0.292929,0,0.00149028,-0.53117,-0.0333333,0
-0.882353,-0.145729,0.0819672,-0.414141,0,-0.207153,-0.766866,-0.666667,1
-0.0588235,0.839196,0.0491803,0,0,-0.305514,-0.492741,-0.633333,0
-0.882353,-0.105528,0.0819672,-0.535354,-0.777778,-0.162444,-0.923997,0,1
0,0.376884,-0.344262,-0.292929,-0.602837,0.28465,0.887276,-0.6,0
-0.411765,0.165829,0.213115,0,0,-0.23696,-0.894962,-0.7,1
-0.647059,-0.21608,-0.180328,-0.353535,-0.791962,-0.0760059,-0.854825,-0.833333,0
0.176471,0.155779,0,0,0,0.052161,-0.952178,-0.733333,1
-0.764706,0.979899,0.147541,-0.0909091,0.283688,-0.0909091,-0.931682,0.0666667,0
-0.0588235,0.256281,0.57377,0,0,0,-0.868488,0.1,0
-0.529412,0.105528,0.508197,0,0,0.120715,-0.903501,-0.7,1
0.176471,0.688442,0.213115,0,0,0.132638,-0.608027,-0.566667,0
0.176471,0.396985,0.311475,0,0,-0.19225,0.163962,0.2,1
-0.882353,0.899497,-0.0163934,-0.535354,1,-0.102832,-0.726729,0.266667,0
-0.176471,0.00502513,0,0,0,-0.105812,-0.653288,-0.633333,0
0,0.18593,0.377049,-0.0505051,-0.456265,0.365127,-0.596072,-0.666667,0
-0.176471,0.0753769,0.213115,0,0,-0.117735,-0.849701,-0.666667,0
-0.882353,0.0351759,-0.508197,-0.232323,-0.803783,0.290611,-0.910333,-0.6,1
-0.882353,0.155779,0.147541,-0.393939,-0.77305,0.0312965,-0.614859,-0.633333,0
-0.647059,0.266332,0.442623,-0.171717,-0.444444,0.171386,-0.465414,-0.8,1
-0.0588235,-0.00502513,0.377049,0,0,0.0551417,-0.735269,-0.0333333,1
-0.176471,0.969849,0.47541,0,0,0.186289,-0.681469,-0.333333,0
0.0588235,0.19598,0.311475,-0.292929,0,-0.135618,-0.842015,-0.733333,0
0.176471,0.256281,0.147541,-0.474747,-0.728132,-0.0730253,-0.891546,-0.333333,0
-0.176471,0.477387,0.245902,0,0,0.174367,-0.847139,-0.266667,0
-0.882353,-0.0251256,0.0819672,-0.69697,-0.669031,-0.308495,-0.650726,-0.966667,1
0.529412,0.457286,0.344262,-0.616162,-0.739953,-0.338301,-0.857387,0.2,1
-0.411765,0.175879,0.508197,0,0,0.0163934,-0.778822,-0.433333,1
-0.411765,0.0954774,0.229508,-0.474747,0,0.0730254,-0.600342,0.3,1
-0.647059,0.58794,0.245902,-0.272727,-0.420804,-0.0581222,-0.33988,-0.766667,0
-0.647059,-0.115578,-0.0491803,-0.777778,-0.87234,-0.260805,-0.838599,-0.966667,1
-0.294118,-0.0753769,0.508197,0,0,-0.406855,-0.906063,-0.766667,1
0.176471,0.226131,0.278689,-0.373737,0,-0.177347,-0.629377,-0.2,1
-0.529412,0.0351759,-0.0163934,-0.333333,-0.546099,-0.28465,-0.241674,-0.6,1
0.294118,0.386935,0.245902,0,0,-0.0104321,-0.707942,-0.533333,1
0.0588235,0.0251256,0.245902,-0.252525,0,-0.019374,-0.498719,-0.166667,0
-0.764706,-0.0954774,0.114754,-0.151515,0,0.138599,-0.637062,-0.8,0
-0.529412,0.115578,0.180328,-0.0505051,-0.510638,0.105812,0.12041,0.166667,0
-0.647059,0.809045,0.0491803,-0.494949,-0.834515,0.0134128,-0.835184,-0.833333,1
-0.176471,0.336683,0.377049,0,0,0.198212,-0.472246,-0.466667,1
-0.176471,0.0653266,0.508197,-0.636364,0,-0.323398,-0.865927,-0.1,1
0.0588235,0.718593,0.803279,-0.515152,-0.432624,0.353204,-0.450897,0.1,0
-0.176471,0.59799,0.0491803,0,0,-0.183308,-0.815542,-0.366667,1
0,0.809045,0.0819672,-0.212121,0,0.251863,0.549957,-0.866667,0
-0.882353,0.467337,-0.0819672,0,0,-0.114754,-0.58497,-0.733333,1
-0.764706,-0.286432,0.147541,-0.454545,0,-0.165425,-0.566183,-0.966667,1
-0.176471,0.0351759,0.0819672,-0.353535,0,0.165425,-0.772844,-0.666667,0
-0.176471,0.0552764,0,0,0,0,-0.806149,-0.9,1
-0.882353,0.0351759,0.311475,-0.777778,-0.806147,-0.421759,-0.64731,-0.966667,1
-0.882353,0.0150754,-0.180328,-0.69697,-0.914894,-0.278688,-0.617421,-0.833333,1
-0.411765,-0.115578,0.0819672,-0.575758,-0.945626,-0.272727,-0.774552,-0.7,1
-0.0588235,0.768844,0.47541,-0.313131,-0.29078,0.004471,-0.667805,0.233333,0
-0.176471,0.507538,0.0819672,-0.151515,-0.191489,0.0342773,-0.453459,-0.3,1
-0.882353,-0.266332,-0.180328,-0.79798,0,-0.314456,-0.854825,0,1
-0.176471,0.879397,0.114754,-0.212121,-0.281324,0.123696,-0.849701,-0.333333,0
0,0.00502513,0.442623,0.212121,-0.739953,0.394933,-0.24509,-0.666667,1
0,0.467337,0.344262,0,0,0.207154,0.454313,-0.233333,1
0,0.0552764,0.0491803,-0.171717,-0.664303,0.23696,-0.918873,-0.966667,1
-0.764706,-0.155779,0,0,0,0,-0.807003,0,1
-0.0588235,0.336683,0.180328,0,0,-0.019374,-0.836038,-0.4,0
-0.411765,-0.557789,0.0163934,0,0,-0.254843,-0.565329,-0.5,1
-0.764706,0.417085,-0.0491803,-0.313131,-0.6974,-0.242921,-0.469684,-0.9,1
-0.176471,0.145729,0.0819672,0,0,-0.0223547,-0.846285,-0.3,0
-0.411765,-0.00502513,0.213115,-0.454545,0,-0.135618,-0.893254,-0.633333,1
0,0.0954774,0.442623,-0.393939,0,-0.0312965,-0.336465,-0.433333,0
-0.764706,0.0954774,0.508197,0,0,0.272727,-0.345004,0.1,1
-0.882353,-0.0452261,0.0819672,-0.737374,-0.910165,-0.415797,-0.781383,-0.866667,1
-0.529412,0.467337,0.393443,-0.454545,-0.763593,-0.138599,-0.905209,-0.8,1
-0.764706,0.00502513,0.0819672,-0.59596,-0.787234,-0.019374,-0.326217,-0.766667,0
-0.411765,0.396985,0.0491803,-0.292929,-0.669031,-0.147541,-0.715628,-0.833333,1
0.529412,0.266332,0.47541,0,0,0.293592,-0.568745,-0.3,0
-0.529412,0.296482,0.409836,-0.59596,-0.361702,0.0461997,-0.869342,-0.933333,1
-0.882353,-0.20603,0.229508,-0.393939,0,-0.0461997,-0.728437,-0.966667,1
-0.882353,0,-0.213115,-0.59596,0,-0.263785,-0.947054,-0.966667,1
-0.176471,-0.376884,0.278689,0,0,-0.028316,-0.732707,-0.333333,1
-0.411765,-0.0452261,0.180328,-0.333333,0,0.123696,-0.75064,-0.8,1
0,0.316583,0,0,0,0.28763,-0.836038,-0.833333,0
-0.764706,0.125628,0.0819672,-0.555556,0,-0.254843,-0.804441,-0.9,1
-0.647059,0.135678,-0.278689,-0.737374,0,-0.33234,-0.947054,-0.966667,1
-0.764706,-0.256281,0,0,0,0,-0.979505,-0.966667,1
-0.176471,-0.165829,0.278689,-0.474747,-0.832151,-0.126677,-0.411614,-0.5,1
0,0.0150754,0.0655738,-0.434343,0,-0.266766,-0.864219,-0.966667,1
-0.411765,0.376884,0.770492,0,0,0.454545,-0.872758,-0.466667,0
-0.764706,0.105528,0.213115,-0.414141,-0.704492,-0.0342771,-0.470538,-0.8,1
0.529412,0.0653266,0.180328,0.0909091,0,0.0909091,-0.914603,-0.2,1
-0.764706,0.00502513,0.114754,-0.494949,-0.832151,0.147541,-0.789923,-0.833333,1
0.764706,0.366834,0.147541,-0.353535,-0.739953,0.105812,-0.935952,-0.266667,0
-0.882353,0.0753769,0.114754,-0.616162,0,-0.210134,-0.925705,-0.9,1
-0.882353,-0.19598,-0.0983607,0,0,-0.4307,-0.846285,0,1
-0.529412,0.236181,0.311475,-0.69697,-0.583924,-0.0461997,-0.688301,-0.566667,1
-0.176471,-0.18593,0.278689,-0.191919,-0.886525,0.391952,-0.843723,-0.3,1
-0.529412,0.346734,0.180328,0,0,-0.290611,-0.83006,0.3,0
-0.764706,0.427136,0.344262,-0.636364,-0.8487,-0.263785,-0.416738,0,1
-0.294118,0.447236,0.180328,-0.454545,-0.460993,0.0104323,-0.848847,-0.366667,1
-0.764706,-0.0753769,0.0163934,-0.434343,0,-0.0581222,-0.955594,-0.9,1
-0.882353,-0.286432,-0.213115,-0.636364,-0.820331,-0.391952,-0.790777,-0.966667,1
-0.294118,-0.0653266,-0.180328,-0.393939,-0.8487,-0.14456,-0.762596,-0.933333,1
-0.882353,0.226131,0.47541,0.030303,-0.479905,0.481371,-0.789069,-0.666667,0
-0.882353,0.638191,0.180328,0,0,0.162444,-0.0230572,-0.6,0
-0.882353,0.517588,-0.0163934,0,0,-0.222057,-0.913749,-0.966667,1
0,0.256281,0.57377,0,0,-0.329359,-0.842869,0,1
-0.882353,-0.18593,0.180328,-0.636364,-0.905437,-0.207153,-0.824936,-0.9,1
-0.764706,-0.145729,0.0655738,0,0,0.180328,-0.272417,-0.8,1
-0.882353,0.266332,-0.0819672,-0.414141,-0.640662,-0.14456,-0.382579,0,1
-0.882353,-0.0351759,1,0,0,-0.33234,-0.889838,-0.8,1
-0.529412,0.447236,-0.0491803,-0.434343,-0.669031,-0.120715,-0.82152,-0.466667,1
-0.647059,-0.165829,-0.0491803,-0.373737,-0.957447,0.0223547,-0.779675,-0.866667,1
0,-0.0452261,0.393443,-0.494949,-0.914894,0.114754,-0.855679,-0.9,0
-0.647059,0.718593,0.180328,-0.333333,-0.680851,-0.00745157,-0.89667,-0.9,0
-0.0588235,0.557789,0.0163934,-0.474747,0.170213,0.0134128,-0.602904,-0.166667,0
-0.882353,-0.105528,0.245902,-0.313131,-0.91253,-0.0700447,-0.902647,-0.933333,1
-0.529412,-0.236181,0.0163934,0,0,0.0134128,-0.732707,-0.866667,1
-0.176471,0.60804,-0.114754,-0.353535,-0.586288,-0.0909091,-0.564475,-0.4,0
-0.529412,0.467337,0.508197,0,0,-0.0700447,-0.606319,0.333333,0
-0.411765,0.246231,0.213115,0,0,0.0134128,-0.878736,-0.433333,0
-0.411765,-0.21608,-0.213115,0,0,0.004471,-0.508113,-0.866667,1
-0.529412,-0.0251256,-0.0163934,-0.535354,0,-0.159463,-0.688301,-0.966667,1
-0.529412,-0.00502513,0.245902,-0.69697,-0.879433,-0.308495,-0.876174,0,1
0,0.628141,0.245902,0.131313,-0.763593,0.585693,-0.418446,-0.866667,0
-0.294118,0.115578,0.0491803,-0.212121,0,0.0193741,-0.844577,-0.9,1
-0.764706,0.0753769,0.213115,-0.393939,-0.763593,0.00149028,-0.721605,-0.933333,1
-0.411765,0.326633,0.311475,0,0,-0.201192,-0.907771,0.6,1
0,0.135678,0.245902,0,0,-0.00745157,-0.829206,-0.933333,0
-0.882353,-0.115578,-0.508197,-0.151515,-0.765957,0.639344,-0.64304,-0.833333,0
-0.647059,0.20603,0.147541,-0.393939,-0.680851,0.278689,-0.680615,-0.7,1
-0.882353,0.18593,-0.0491803,-0.272727,-0.777778,-0.00745157,-0.843723,-0.933333,1
-0.882353,0.175879,0.442623,-0.515152,-0.65721,0.028316,-0.722459,-0.366667,0
0,0.0552764,0.377049,0,0,-0.168405,-0.433817,0.366667,0
-0.529412,0.738693,0.147541,-0.717172,-0.602837,-0.114754,-0.758326,-0.6,0
0.0588235,0.226131,-0.0819672,0,0,-0.00745157,-0.115286,-0.6,0
-0.647059,0.708543,0.0491803,-0.252525,-0.468085,0.028316,-0.762596,-0.7,0
-0.0588235,-0.155779,0.213115,-0.373737,0,0.14158,-0.676345,-0.4,1
-0.764706,-0.0351759,0.114754,-0.737374,-0.884161,-0.371088,-0.514091,-0.833333,1
-0.764706,0.256281,-0.0163934,-0.59596,-0.669031,0.00745157,-0.99146,-0.666667,1
0,0.00502513,0.147541,-0.474747,-0.881797,-0.0819672,-0.556789,0,1
0,-0.0653266,-0.0163934,-0.494949,-0.782506,-0.14456,-0.612297,-0.966667,1
0,0.296482,0.311475,0,0,-0.0700447,-0.466268,-0.733333,1
-0.411765,0.0552764,0.180328,-0.414141,-0.231678,0.0998511,-0.930828,-0.766667,1
-0.647059,0.286432,0.278689,0,0,-0.371088,-0.837746,0.133333,1
-0.411765,0.0653266,0.344262,-0.393939,0,0.177347,-0.822374,-0.433333,1
-0.764706,0.0854271,-0.147541,-0.474747,-0.851064,-0.0312965,-0.795047,-0.966667,1
0.176471,0.0854271,0.0819672,0,0,-0.0342771,-0.83433,-0.3,0
-0.529412,0.547739,0.0163934,-0.373737,-0.328605,-0.0223547,-0.864219,-0.933333,1
0,0.0251256,0.229508,-0.535354,0,0,-0.578138,0,1
0.0588235,-0.427136,0.311475,-0.252525,0,-0.0223547,-0.984629,-0.333333,1
-0.764706,0.0653266,0.0491803,-0.292929,-0.718676,-0.0909091,0.12895,-0.566667,1
-0.411765,0.477387,0.278689,0,0,0.004471,-0.880444,0.466667,1
-0.764706,-0.0954774,0.147541,-0.656566,0,-0.186289,-0.994022,-0.966667,1
-0.882353,0.366834,0.213115,0.010101,-0.51773,0.114754,-0.725875,-0.9,1
-0.529412,0.145729,0.0655738,0,0,-0.347243,-0.697694,-0.466667,1
0.0588235,0.567839,0.409836,-0.434343,-0.63357,0.0223547,-0.0512383,-0.3,0
-0.882353,0.537688,0.344262,-0.151515,0.146572,0.210134,-0.479932,-0.933333,1
-0.0588235,0.889447,0.278689,0,0,0.42772,-0.949616,-0.266667,0
-0.176471,0.527638,0.442623,-0.111111,0,0.490313,-0.778822,-0.5,0
-0.764706,-0.00502513,-0.147541,-0.69697,-0.777778,-0.266766,-0.52263,0,1
-0.882353,0.0954774,-0.0819672,-0.575758,-0.680851,-0.248882,-0.355252,-0.933333,1
-0.764706,-0.115578,0.213115,-0.616162,-0.874704,-0.135618,-0.87105,-0.966667,1
1,0.638191,0.180328,-0.171717,-0.730496,0.219076,-0.368915,-0.133333,0
-0.529412,0.517588,0.47541,-0.232323,0,-0.114754,-0.815542,-0.5,1
-0.176471,0.0251256,0.213115,-0.191919,-0.751773,0.108793,-0.8924,-0.2,1
0,0.145729,0.311475,-0.313131,-0.326241,0.317437,-0.923997,-0.8,1
-0.764706,0.00502513,0.0491803,-0.535354,0,-0.114754,-0.752348,0,1
0,0.316583,0.442623,0,0,-0.0581222,-0.432109,-0.633333,0
-0.294118,0.0452261,0.213115,-0.636364,-0.631206,-0.108793,-0.450043,-0.333333,0
-0.647059,0.487437,0.0819672,-0.494949,0,-0.0312965,-0.847993,-0.966667,1
-0.529412,0.20603,0.114754,0,0,-0.117735,-0.461144,-0.566667,1
-0.529412,0.105528,0.0819672,0,0,-0.0491803,-0.664389,-0.733333,1
-0.647059,0.115578,0.47541,-0.757576,-0.815603,-0.153502,-0.643894,-0.733333,1
-0.294118,0.0251256,0.344262,0,0,-0.0819672,-0.912895,-0.5,0
-0.294118,0.346734,0.147541,-0.535354,-0.692671,0.0551417,-0.603757,-0.733333,0
-0.764706,-0.125628,0,-0.535354,0,-0.138599,-0.40649,-0.866667,1
-0.882353,-0.20603,-0.0163934,-0.151515,-0.886525,0.296572,-0.487617,-0.933333,1
-0.764706,-0.246231,0.0491803,-0.515152,-0.869976,-0.114754,-0.75064,-0.6,1
-0.0588235,0.798995,0.180328,-0.151515,-0.692671,-0.0253353,-0.452605,-0.5,0
-0.294118,-0.145729,0.278689,0,0,-0.0700447,-0.740393,-0.3,1
0,0.296482,0.803279,-0.0707071,-0.692671,1,-0.794193,-0.833333,0
-0.411765,0.437186,0.278689,0,0,0.341282,-0.904355,-0.133333,1
-0.411765,0.306533,0.344262,0,0,0.165425,-0.250213,-0.466667,0
-0.294118,-0.125628,0.311475,0,0,-0.308495,-0.994876,-0.633333,1
0,0.19598,0.0491803,-0.636364,-0.782506,0.0402385,-0.447481,-0.933333,1
-0.882353,0,0.213115,-0.59596,-0.945626,-0.174367,-0.811272,0,1
-0.411765,-0.266332,-0.0163934,0,0,-0.201192,-0.837746,-0.8,1
-0.529412,0.417085,0.213115,0,0,-0.177347,-0.858241,-0.366667,1
-0.176471,0.949749,0.114754,-0.434343,0,0.0700448,-0.430401,-0.333333,0
-0.0588235,0.819095,0.114754,-0.272727,0.170213,-0.102832,-0.541418,0.3,0
-0.882353,0.286432,0.606557,-0.171717,-0.862884,-0.0461997,0.0614859,-0.6,0
-0.0588235,0.0954774,0.245902,-0.212121,-0.730496,-0.168405,-0.520068,-0.666667,0
-0.411765,0.396985,0.311475,-0.292929,-0.621749,-0.0581222,-0.758326,-0.866667,0
-0.647059,0.115578,0.0163934,0,0,-0.326379,-0.945346,0,1
0.0588235,0.236181,0.147541,-0.111111,-0.777778,-0.0134128,-0.747225,-0.366667,1
-0.176471,0.59799,0.0819672,0,0,-0.0938897,-0.739539,-0.5,0
0.294118,0.356784,0,0,0,0.558867,-0.573015,-0.366667,0
-0.0588235,-0.145729,-0.0983607,-0.59596,0,-0.272727,-0.95047,-0.3,1
-0.411765,0.58794,0.377049,-0.171717,-0.503546,0.174367,-0.729291,-0.733333,0
-0.882353,0.0552764,-0.0491803,0,0,-0.275708,-0.906917,0,1
-0.647059,0.0753769,0.0163934,-0.737374,-0.886525,-0.317437,-0.487617,-0.933333,0
-0.529412,0.0954774,0.0491803,-0.111111,-0.765957,0.0372578,-0.293766,-0.833333,0
-0.529412,0.487437,-0.0163934,-0.454545,-0.248227,-0.0789866,-0.938514,-0.733333,0
0,0.135678,0.311475,-0.676768,0,-0.0760059,-0.320239,0,1
-0.882353,0.386935,0.344262,0,0,0.195231,-0.865073,-0.766667,1
0,0.0854271,0.114754,-0.59596,0,-0.186289,-0.394535,-0.633333,1
-0.764706,-0.00502513,0.147541,-0.676768,-0.895981,-0.391952,-0.865927,-0.8,1
-0.294118,0.0351759,0.180328,-0.353535,-0.550827,0.123696,-0.789923,0.133333,1
-0.411765,0.115578,0.180328,-0.434343,0,-0.28763,-0.719044,-0.8,1
-0.0588235,0.969849,0.245902,-0.414141,-0.338061,0.117735,-0.549957,0.2,0
-0.411765,0.628141,0.704918,0,0,0.123696,-0.93766,0.0333333,0
-0.882353,-0.0351759,0.0491803,-0.454545,-0.794326,-0.0104321,-0.819812,0,1
-0.176471,0.849246,0.377049,-0.333333,0,0.0581222,-0.76345,-0.333333,0
-0.764706,-0.18593,-0.0163934,-0.555556,0,-0.174367,-0.818958,-0.866667,1
0,0.477387,0.393443,0.0909091,0,0.275708,-0.746371,-0.9,1
-0.176471,0.798995,0.557377,-0.373737,0,0.0193741,-0.926558,0.3,1
0,0.407035,0.0655738,-0.474747,-0.692671,0.269747,-0.698548,-0.9,0
0.0588235,0.125628,0.344262,-0.353535,-0.586288,0.0193741,-0.844577,-0.5,0
0.411765,0.517588,0.147541,-0.191919,-0.359338,0.245902,-0.432963,-0.433333,0
-0.411765,0.0954774,0.0163934,-0.171717,-0.695035,0.0670641,-0.627669,-0.866667,0
-0.294118,0.256281,0.114754,-0.393939,-0.716312,-0.105812,-0.670367,-0.633333,1
-0.411765,-0.145729,0.213115,-0.555556,0,-0.135618,-0.0213493,-0.633333,0
-0.411765,0.125628,0.0819672,0,0,0.126677,-0.843723,-0.333333,0
0,0.778894,-0.0163934,-0.414141,0.130024,0.0312965,-0.151153,0,0
-0.764706,0.58794,0.47541,0,0,-0.0581222,-0.379163,0.5,0
-0.176471,0.19598,0,0,0,-0.248882,-0.88813,-0.466667,1
-0.176471,0.427136,-0.0163934,-0.333333,-0.550827,-0.14158,-0.479932,0.333333,1
-0.882353,0.00502513,0.0819672,-0.69697,-0.867612,-0.296572,-0.497865,-0.833333,1
-0.882353,-0.125628,0.278689,-0.454545,-0.92435,0.0312965,-0.980359,-0.966667,1
0,0.0150754,0.245902,0,0,0.0640835,-0.897523,-0.833333,1
-0.647059,0.628141,-0.147541,-0.232323,0,0.108793,-0.509821,-0.9,0
-0.529412,0.979899,0.147541,-0.212121,0.758865,0.0938898,0.922289,-0.666667,1
0,0.175879,0.311475,-0.373737,-0.874704,0.347243,-0.990606,-0.9,1
-0.529412,0.427136,0.409836,0,0,0.311475,-0.515798,-0.966667,0
-0.294118,0.346734,0.311475,-0.252525,-0.125296,0.377049,-0.863365,-0.166667,0
-0.882353,-0.20603,0.311475,-0.494949,-0.91253,-0.242921,-0.568745,-0.966667,1
-0.529412,0.226131,0.114754,0,0,0.0432191,-0.730145,-0.733333,1
-0.647059,-0.256281,0.114754,-0.434343,-0.893617,-0.114754,-0.816396,-0.933333,1
-0.529412,0.718593,0.180328,0,0,0.299553,-0.657558,-0.833333,0
0,0.798995,0.47541,-0.454545,0,0.314456,-0.480786,-0.933333,0
0.0588235,0.648241,0.377049,-0.575758,0,-0.0819672,-0.35696,-0.633333,0
0,0.0452261,0.245902,0,0,-0.451565,-0.569599,-0.8,1
-0.882353,-0.0854271,0.0491803,-0.515152,0,-0.129657,-0.902647,0,1
-0.529412,-0.0854271,0.147541,-0.353535,-0.791962,-0.0134128,-0.685739,-0.966667,1
-0.647059,0.396985,-0.114754,0,0,-0.23696,-0.723313,-0.966667,0
-0.294118,0.19598,-0.180328,-0.555556,-0.583924,-0.19225,0.058924,-0.6,0
-0.764706,0.467337,0.245902,-0.292929,-0.541371,0.138599,-0.785653,-0.733333,1
0.0588235,0.849246,0.393443,-0.69697,0,-0.105812,-0.030743,-0.0666667,0
0.176471,0.226131,0.114754,0,0,-0.0700447,-0.846285,-0.333333,1
0,0.658291,0.47541,-0.333333,0.607565,0.558867,-0.701964,-0.933333,1
0.0588235,0.246231,0.147541,-0.333333,-0.0496454,0.0551417,-0.82579,-0.566667,1
-0.882353,0.115578,0.409836,-0.616162,0,-0.102832,-0.944492,-0.933333,1
0.0588235,0.0653266,-0.147541,0,0,-0.0700447,-0.742101,-0.3,1
-0.764706,0.296482,0.377049,0,0,-0.165425,-0.824082,-0.8,1
-0.764706,-0.0954774,0.311475,-0.717172,-0.869976,-0.272727,-0.853971,-0.9,1
0,-0.135678,0.114754,-0.353535,0,0.0670641,-0.863365,-0.866667,1
0.411765,-0.0753769,0.0163934,-0.858586,-0.390071,-0.177347,-0.275833,-0.233333,0
-0.882353,0.135678,0.0491803,-0.292929,0,0.00149028,-0.602904,0,0
-0.647059,0.115578,-0.0819672,-0.212121,0,-0.102832,-0.590948,-0.7,1
-0.764706,0.145729,0.114754,-0.555556,0,-0.14456,-0.988044,-0.866667,1
-0.882353,0.939698,-0.180328,-0.676768,-0.113475,-0.228018,-0.507259,-0.9,1
-0.647059,0.919598,0.114754,-0.69697,-0.692671,-0.0789866,-0.811272,-0.566667,1
-0.647059,0.417085,0,0,0,-0.105812,-0.416738,-0.8,0
-0.529412,-0.0452261,0.147541,-0.353535,0,-0.0432191,-0.54398,-0.9,1
-0.647059,0.427136,0.311475,-0.69697,0,-0.0342771,-0.895816,0.4,1
-0.529412,0.236181,0.0163934,0,0,-0.0461997,-0.873612,-0.533333,0
-0.411765,-0.0351759,0.213115,-0.636364,-0.841608,0.00149028,-0.215201,-0.266667,1
0,0.386935,0,0,0,0.0819672,-0.269855,-0.866667,0
-0.764706,0.286432,0.0491803,-0.151515,0,0.19225,-0.126388,-0.9,1
0,0.0251256,-0.147541,0,0,-0.251863,0,0,1
-0.764706,0.467337,0,0,0,-0.180328,-0.861657,-0.766667,0
0.176471,0.0150754,0.409836,-0.252525,0,0.359165,-0.0964987,-0.433333,0
-0.764706,0.0854271,0.0163934,-0.353535,-0.867612,-0.248882,-0.957301,0,1
-0.647059,0.226131,0.278689,0,0,-0.314456,-0.849701,-0.366667,1
-0.882353,-0.286432,0.278689,0.010101,-0.893617,-0.0104321,-0.706234,0,1
0.529412,0.0653266,0.147541,0,0,0.0193741,-0.852263,0.0333333,1
-0.764706,0.00502513,0.147541,0.0505051,-0.865248,0.207154,-0.488471,-0.866667,1
-0.176471,0.0653266,-0.0163934,-0.515152,0,-0.210134,-0.813834,-0.733333,0
0,0.0452261,0.0491803,-0.535354,-0.725768,-0.171386,-0.678907,-0.933333,1
-0.411765,0.145729,0.213115,0,0,-0.257824,-0.431255,0.2,1
-0.764706,0.0854271,0.0163934,-0.79798,-0.34279,-0.245902,-0.314261,-0.966667,1
0,0.467337,0.147541,0,0,0.129657,-0.781383,-0.766667,0
0.176471,0.296482,0.245902,-0.434343,-0.711584,0.0700448,-0.827498,-0.4,1
-0.176471,0.336683,0.442623,-0.69697,-0.63357,-0.0342771,-0.842869,-0.466667,1
-0.176471,0.61809,0.409836,0,0,-0.0938897,-0.925705,-0.133333,0
-0.764706,0.0854271,0.311475,0,0,-0.195231,-0.845431,0.0333333,0
-0.411765,0.557789,0.377049,-0.111111,0.288416,0.153502,-0.538002,-0.566667,1
-0.882353,0.19598,0.409836,-0.212121,-0.479905,0.359165,-0.376601,-0.733333,0
-0.529412,-0.0351759,-0.0819672,-0.656566,-0.884161,-0.38003,-0.77626,-0.833333,1
-0.411765,0.0854271,0.180328,-0.131313,-0.822695,0.0760059,-0.842015,-0.6,1
0,-0.21608,0.442623,-0.414141,-0.905437,0.0998511,-0.695986,0,1
0,0.0753769,0.0163934,-0.393939,-0.825059,0.0909091,-0.420154,-0.866667,0
-0.764706,0.286432,0.278689,-0.252525,-0.56974,0.290611,-0.0213493,-0.666667,0
-0.882353,0.286432,-0.213115,-0.0909091,-0.541371,0.207154,-0.543126,-0.9,0
0,0.61809,-0.180328,0,0,-0.347243,-0.849701,0.466667,1
-0.294118,0.517588,0.0163934,-0.373737,-0.716312,0.0581222,-0.475662,-0.766667,1
-0.764706,0.467337,0.147541,-0.232323,-0.148936,-0.165425,-0.778822,-0.733333,0
0,0.266332,0.377049,-0.414141,-0.491726,-0.0849478,-0.622545,-0.9,1
0.647059,0.00502513,0.278689,-0.494949,-0.565012,0.0909091,-0.714774,-0.166667,0
-0.0588235,0.125628,0.180328,0,0,-0.296572,-0.349274,0.233333,1
0,0.678392,0,0,0,-0.0372578,-0.350128,-0.7,0
-0.764706,0.447236,-0.0491803,-0.333333,-0.680851,-0.0581222,-0.706234,-0.866667,0
-0.411765,-0.226131,0.344262,-0.171717,-0.900709,0.0670641,-0.93339,-0.533333,1
-0.411765,0.155779,0.606557,0,0,0.576751,-0.88813,-0.766667,0
-0.647059,0.507538,0.245902,0,0,-0.374069,-0.889838,-0.466667,1
-0.764706,0.20603,0.245902,-0.252525,-0.751773,0.183309,-0.883006,-0.733333,1
0.176471,0.61809,0.114754,-0.535354,-0.687943,-0.23994,-0.788215,-0.133333,0
0,0.376884,0.114754,-0.717172,-0.650118,-0.260805,-0.944492,0,1
0,0.286432,0.114754,-0.616162,-0.574468,-0.0909091,0.121264,-0.866667,0
-0.764706,0.246231,0.114754,-0.434343,-0.515366,-0.019374,-0.319385,-0.7,0
-0.294118,-0.19598,0.0819672,-0.393939,0,-0.219076,-0.799317,-0.333333,1
0,0.0653266,0.147541,-0.252525,-0.650118,0.174367,-0.549957,-0.966667,1
-0.764706,0.557789,0.213115,-0.656566,-0.77305,-0.207153,-0.69684,-0.8,0
-0.647059,0.135678,-0.180328,-0.79798,-0.799054,-0.120715,-0.532024,-0.866667,1
-0.176471,0.0954774,0.311475,-0.373737,0,0.0700448,-0.104184,-0.266667,0
-0.764706,0.125628,0.114754,-0.555556,-0.777778,0.0163934,-0.797609,-0.833333,1
-0.647059,-0.00502513,0.311475,-0.777778,-0.8487,-0.424739,-0.824082,-0.7,1
-0.647059,0.829146,0.213115,0,0,-0.0909091,-0.77199,-0.733333,0
-0.647059,0.155779,0.0819672,-0.212121,-0.669031,0.135618,-0.938514,-0.766667,1
-0.294118,0.949749,0.278689,0,0,-0.299553,-0.956447,0.266667,0
-0.529412,0.296482,-0.0163934,-0.757576,-0.453901,-0.180328,-0.616567,-0.666667,1
-0.647059,0.125628,0.213115,-0.393939,0,-0.0581222,-0.898377,-0.866667,0
0,0.246231,0.147541,-0.59596,0,-0.183308,-0.849701,-0.5,0
0.529412,0.527638,0.47541,-0.333333,-0.931442,-0.201192,-0.442357,-0.266667,0
-0.764706,0.125628,0.229508,-0.353535,0,0.0640835,-0.940222,0,1
-0.882353,0.577889,0.180328,-0.575758,-0.602837,-0.23696,-0.961571,-0.9,1
-0.882353,0.226131,0.0491803,-0.353535,-0.631206,0.0461997,-0.475662,-0.7,0
0.176471,0.798995,0.147541,0,0,0.0461997,-0.895816,-0.466667,1
-0.764706,0.0251256,0.409836,-0.272727,-0.716312,0.356185,-0.958155,-0.933333,0
-0.294118,0.0552764,0.147541,-0.353535,-0.839243,-0.0819672,-0.962425,-0.466667,1
-0.0588235,0.18593,0.180328,-0.616162,0,-0.311475,0.193851,-0.166667,1
-0.764706,-0.125628,-0.0491803,-0.676768,-0.877069,-0.0253353,-0.924851,-0.866667,1
-0.882353,0.809045,0,0,0,0.290611,-0.82579,-0.333333,0
0.411765,0.0653266,0.311475,0,0,-0.296572,-0.949616,-0.233333,1
-0.882353,-0.0452261,-0.0163934,-0.636364,-0.862884,-0.28763,-0.844577,-0.966667,1
0,0.658291,0.245902,-0.131313,-0.397163,0.42772,-0.845431,-0.833333,1
0,0.175879,0,0,0,0.00745157,-0.270709,-0.233333,1
-0.411765,0.155779,0.245902,0,0,-0.0700447,-0.773698,-0.233333,0
0.0588235,0.527638,0.278689,-0.313131,-0.595745,0.0193741,-0.304014,-0.6,0
-0.176471,0.788945,0.377049,0,0,0.18927,-0.783945,-0.333333,0
-0.882353,0.306533,0.147541,-0.737374,-0.751773,-0.228018,-0.663535,-0.966667,1
-0.882353,-0.0452261,0.213115,-0.575758,-0.827423,-0.228018,-0.491887,-0.5,1
-0.882353,0,0.114754,-0.292929,0,-0.0461997,-0.734415,-0.966667,1
-0.411765,0.226131,0.409836,0,0,0.0342773,-0.818958,-0.6,1
-0.0588235,-0.0452261,0.180328,0,0,0.0968703,-0.652434,0.2,1
-0.0588235,0.266332,0.442623,-0.272727,-0.744681,0.147541,-0.768574,-0.0666667,1
-0.882353,0.396985,-0.245902,-0.616162,-0.803783,-0.14456,-0.508113,-0.966667,1
-0.647059,0.165829,0,0,0,-0.299553,-0.906917,-0.933333,1
-0.647059,-0.00502513,0.0163934,-0.616162,-0.825059,-0.350224,-0.828352,-0.833333,1
-0.411765,0,0.311475,-0.353535,0,0.222057,-0.771136,-0.466667,0
-0.529412,-0.0753769,0.311475,0,0,0.257824,-0.864219,-0.733333,1
-0.529412,0.376884,0.377049,0,0,-0.0700447,-0.851409,-0.7,1
-0.647059,-0.386935,0.344262,-0.434343,0,0.0253354,-0.859095,-0.166667,1
-0.882353,-0.0954774,0.0163934,-0.757576,-0.898345,-0.18927,-0.571307,-0.9,1
-0.647059,-0.0954774,0.278689,0,0,0.272727,-0.58924,0,1
0.0588235,0.658291,0.442623,0,0,-0.0938897,-0.808711,-0.0666667,0
-0.882353,0.256281,-0.180328,-0.191919,-0.605201,-0.00745157,-0.24509,-0.766667,0
0.529412,0.296482,0,-0.393939,0,0.18927,-0.5807,-0.233333,0
0.411765,-0.115578,0.213115,-0.191919,-0.87234,0.052161,-0.743809,-0.1,1
-0.882353,0.969849,0.245902,-0.272727,-0.411348,0.0879285,-0.319385,-0.733333,0
-0.411765,0.899497,0.0491803,-0.333333,-0.231678,-0.0700447,-0.568745,-0.733333,0
-0.411765,0.58794,0.147541,0,0,-0.111773,-0.889838,0.4,1
-0.411765,0.0351759,0.770492,-0.252525,0,0.168405,-0.806149,0.466667,1
-0.529412,0.467337,0.278689,0,0,0.147541,-0.622545,0.533333,0
-0.529412,0.477387,0.213115,-0.494949,-0.307329,0.0402385,-0.737831,-0.7,1
-0.411765,-0.00502513,-0.114754,-0.434343,-0.803783,0.0134128,-0.640478,-0.7,1
-0.294118,0.246231,0.180328,0,0,-0.177347,-0.752348,-0.733333,0
0,0.0150754,0.0491803,-0.656566,0,-0.374069,-0.851409,0,1
-0.647059,-0.18593,0.409836,-0.676768,-0.843972,-0.180328,-0.805295,-0.966667,1
-0.882353,0.336683,0.672131,-0.434343,-0.669031,-0.0223547,-0.866781,-0.2,0
-0.647059,0.738693,0.344262,-0.030303,0.0992908,0.14456,0.758326,-0.866667,0
0,0.18593,0.0491803,-0.535354,-0.789598,0,0.411614,0,1
0,-0.155779,0.0491803,-0.555556,-0.843972,0.0670641,-0.601196,0,1
-0.764706,0.0552764,-0.0491803,-0.191919,-0.777778,0.0402385,-0.874466,-0.866667,1
-0.764706,0.226131,-0.147541,-0.131313,-0.626478,0.0789866,-0.369769,-0.766667,1
0.411765,0.407035,0.344262,-0.131313,-0.231678,0.168405,-0.615713,0.233333,0
0,-0.0150754,0.344262,-0.69697,-0.801418,-0.248882,-0.811272,-0.966667,1
-0.882353,-0.125628,-0.0163934,-0.252525,-0.822695,0.108793,-0.631939,-0.966667,1
-0.529412,0.567839,0.229508,0,0,0.439642,-0.863365,-0.633333,0
0,-0.0653266,0.639344,-0.212121,-0.829787,0.293592,-0.194705,-0.533333,1
-0.882353,0.0753769,0.180328,-0.393939,-0.806147,-0.0819672,-0.3655,-0.9,1
0,0.0552764,0.114754,-0.555556,0,-0.403875,-0.865073,-0.966667,1
-0.882353,0.0954774,-0.0163934,-0.838384,-0.56974,-0.242921,-0.257899,0,1
-0.882353,-0.0954774,0.0163934,-0.636364,-0.86052,-0.251863,0.0162254,-0.866667,1
-0.882353,0.256281,0.147541,-0.515152,-0.739953,-0.275708,-0.877882,-0.866667,1
-0.882353,0.19598,-0.114754,-0.737374,-0.881797,-0.33532,-0.891546,-0.9,1
-0.411765,0.165829,0.213115,-0.414141,0,-0.0372578,-0.502989,-0.533333,0
-0.0588235,0.0552764,0.639344,-0.272727,0,0.290611,-0.862511,-0.2,0
-0.411765,0.447236,0.344262,-0.474747,-0.326241,-0.0461997,-0.680615,0.233333,0
-0.647059,0.00502513,0.114754,-0.535354,-0.808511,-0.0581222,-0.256191,-0.766667,1
-0.882353,0.00502513,0.0819672,-0.414141,-0.536643,-0.0461997,-0.687447,-0.3,1
-0.411765,0.668342,0.245902,0,0,0.362146,-0.77626,-0.8,0
-0.882353,0.316583,0.0491803,-0.717172,-0.0189125,-0.293592,-0.734415,0,1
-0.529412,0.165829,0.180328,-0.757576,-0.794326,-0.341282,-0.671221,-0.466667,1
-0.529412,0.58794,0.278689,0,0,-0.019374,-0.380871,-0.666667,0
-0.764706,0.276382,-0.0491803,-0.515152,-0.349882,-0.174367,0.299744,-0.866667,1
-0.647059,-0.0351759,-0.0819672,-0.313131,-0.728132,-0.263785,-0.260461,-0.4,1
0,0.316583,0.0819672,-0.191919,0,0.0223547,-0.899231,-0.966667,0
-0.647059,-0.175879,0.147541,0,0,-0.371088,-0.734415,-0.866667,1
-0.647059,0.939698,0.147541,-0.373737,0,0.0402385,-0.860803,-0.866667,0
-0.529412,-0.0452261,0.0491803,0,0,-0.0461997,-0.92912,-0.666667,0
-0.411765,0.366834,0.377049,-0.171717,-0.791962,0.0432191,-0.822374,-0.533333,0
0.0588235,-0.276382,0.278689,-0.494949,0,-0.0581222,-0.827498,-0.433333,1
-0.411765,0.688442,0.0491803,0,0,-0.019374,-0.951324,-0.333333,0
-0.764706,0.236181,-0.213115,-0.353535,-0.609929,0.254843,-0.622545,-0.833333,1
-0.529412,0.155779,0.180328,0,0,-0.138599,-0.745517,-0.166667,0
0,0.0150754,0.0163934,0,0,-0.347243,-0.779675,-0.866667,1
-0.0588235,0.979899,0.213115,0,0,-0.228018,-0.0495303,-0.4,0
-0.882353,0.728643,0.114754,-0.010101,0.368794,0.263785,-0.467122,-0.766667,0
-0.294118,0.0251256,0.47541,-0.212121,0,0.0640835,-0.491033,-0.766667,1
-0.882353,0.125628,0.180328,-0.393939,-0.583924,0.0253354,-0.615713,-0.866667,1
-0.882353,0.437186,0.377049,-0.535354,-0.267139,0.263785,-0.147737,-0.966667,1
-0.882353,0.437186,0.213115,-0.555556,-0.855792,-0.219076,-0.847993,0,1
0,0.386935,-0.0163934,-0.292929,-0.605201,0.0312965,-0.610589,0,0
-0.647059,0.738693,0.377049,-0.333333,0.120567,0.0640835,-0.846285,-0.966667,0
-0.882353,-0.0251256,0.114754,-0.575758,0,-0.18927,-0.131512,-0.966667,1
-0.529412,0.447236,0.344262,-0.353535,0,0.147541,-0.59351,-0.466667,0
-0.882353,-0.165829,0.114754,0,0,-0.457526,-0.533732,-0.8,1
-0.647059,0.296482,0.0491803,-0.414141,-0.728132,-0.213115,-0.87959,-0.766667,0
-0.882353,0.19598,0.442623,-0.171717,-0.598109,0.350224,-0.633646,-0.833333,1
-0.764706,-0.0552764,0.114754,-0.636364,-0.820331,-0.225037,-0.587532,0,1
0,0.0251256,0.0491803,-0.0707071,-0.815603,0.210134,-0.64304,0,1
-0.764706,0.155779,0.0491803,-0.555556,0,-0.0819672,-0.707088,0,1
-0.0588235,0.517588,0.278689,-0.353535,-0.503546,0.278689,-0.625961,-0.5,0
-0.529412,0.849246,0.278689,-0.212121,-0.345154,0.102832,-0.841161,-0.666667,0
0,-0.0552764,0,0,0,0,-0.847993,-0.866667,1
-0.882353,0.819095,0.0491803,-0.393939,-0.574468,0.0163934,-0.786507,-0.433333,0
0,0.356784,0.540984,-0.0707071,-0.65721,0.210134,-0.824082,-0.833333,1
-0.882353,-0.0452261,0.344262,-0.494949,-0.574468,0.0432191,-0.867635,-0.266667,0
-0.764706,-0.00502513,0,0,0,-0.338301,-0.974381,-0.933333,1
-0.647059,-0.105528,0.213115,-0.676768,-0.799054,-0.0938897,-0.596072,-0.433333,1
-0.882353,-0.19598,0.213115,-0.777778,-0.858156,-0.105812,-0.616567,-0.966667,1
-0.764706,0.396985,0.229508,0,0,-0.23696,-0.923997,-0.733333,1
-0.882353,-0.0954774,0.114754,-0.838384,0,-0.269747,-0.0947908,-0.5,1
0,0.417085,0,0,0,0.263785,-0.891546,-0.733333,0
0.411765,0.407035,0.393443,-0.333333,0,0.114754,-0.858241,-0.333333,1
-0.411765,0.477387,0.229508,0,0,-0.108793,-0.695986,-0.766667,1
-0.882353,-0.0251256,0.147541,-0.69697,0,-0.457526,-0.941076,0,1
-0.294118,0.0753769,0.442623,0,0,0.0968703,-0.445773,-0.666667,1
0,0.899497,0.704918,-0.494949,0,0.0223547,-0.695132,-0.333333,0
-0.764706,-0.165829,0.0819672,-0.535354,-0.881797,-0.0402384,-0.642186,-0.966667,1
-0.529412,0.175879,0.0491803,-0.454545,-0.716312,-0.0104321,-0.870196,-0.9,1
-0.0588235,0.0854271,0.147541,0,0,-0.0909091,-0.251067,-0.6,0
-0.529412,0.175879,0.0163934,-0.757576,0,-0.114754,-0.742101,-0.7,0
0,0.809045,0.278689,0.272727,-0.966903,0.770492,1,-0.866667,0
-0.882353,0.00502513,0.180328,-0.757576,-0.834515,-0.245902,-0.504697,-0.766667,1
0,-0.0452261,0.311475,-0.0909091,-0.782506,0.0879285,-0.784799,-0.833333,1
0,0.0452261,0.0491803,-0.252525,-0.8487,0.00149028,-0.631085,-0.966667,0
0,0.20603,0.213115,-0.636364,-0.851064,-0.0909091,-0.823228,-0.833333,1
-0.882353,-0.175879,0.0491803,-0.737374,-0.775414,-0.368107,-0.712212,-0.933333,1
-0.764706,0.346734,0.147541,0,0,-0.138599,-0.603757,-0.933333,0
0,-0.0854271,0.114754,-0.353535,-0.503546,0.18927,-0.741247,-0.866667,1
-0.764706,0.19598,0,0,0,-0.415797,-0.356106,0.7,1
-0.764706,0.00502513,-0.114754,-0.434343,-0.751773,0.126677,-0.641332,-0.9,1
0.647059,0.758794,0.0163934,-0.393939,0,0.00149028,-0.885568,-0.433333,0
-0.882353,0.356784,-0.114754,0,0,-0.204173,-0.479932,0.366667,1
-0.411765,-0.135678,0.114754,-0.434343,-0.832151,-0.0998509,-0.755764,-0.9,1
0.0588235,0.346734,0.213115,-0.333333,-0.858156,-0.228018,-0.673783,1,1
0.0588235,0.20603,0.180328,-0.555556,-0.867612,-0.38003,-0.440649,-0.1,1
-0.882353,-0.286432,0.0163934,0,0,-0.350224,-0.711358,-0.833333,1
-0.0588235,-0.256281,0.147541,-0.191919,-0.884161,0.052161,-0.46456,-0.4,1
-0.411765,-0.115578,0.278689,-0.393939,0,-0.177347,-0.846285,-0.466667,1
0.176471,0.155779,0.606557,0,0,-0.28465,-0.193851,-0.566667,1
0,0.246231,-0.0819672,-0.737374,-0.751773,-0.350224,-0.680615,0,1
0,-0.256281,-0.147541,-0.79798,-0.914894,-0.171386,-0.836892,-0.966667,1
0,-0.0251256,0.0491803,-0.272727,-0.763593,0.0968703,-0.554227,-0.866667,1
-0.0588235,0.20603,0,0,0,-0.105812,-0.910333,-0.433333,0
-0.294118,0.547739,0.278689,-0.171717,-0.669031,0.374069,-0.578992,-0.8,1
-0.882353,0.447236,0.344262,-0.191919,0,0.230999,-0.548249,-0.766667,1
0,0.376884,0.147541,-0.232323,0,-0.0104321,-0.921435,-0.966667,1
0,0.19598,0.0819672,-0.454545,0,0.156483,-0.845431,-0.966667,1
-0.176471,0.366834,0.47541,0,0,-0.108793,-0.887276,-0.0333333,1
-0.529412,0.145729,0.0491803,0,0,-0.138599,-0.959009,-0.9,1
0,0.376884,0.377049,-0.454545,0,-0.186289,-0.869342,0.266667,1
-0.764706,0.0552764,0.311475,-0.0909091,-0.548463,0.004471,-0.459436,-0.733333,0
-0.176471,0.145729,0.245902,-0.656566,-0.739953,-0.290611,-0.668659,-0.666667,1
-0.0588235,0.266332,0.213115,-0.232323,-0.822695,-0.228018,-0.928266,-0.4,1
-0.529412,0.326633,0.409836,-0.373737,0,-0.165425,-0.708796,0.4,1
-0.647059,0.58794,0.147541,-0.393939,-0.224586,0.0581222,-0.772844,-0.533333,0
0,0.236181,0.442623,-0.252525,0,0.0491804,-0.898377,-0.733333,1
-0.529412,-0.145729,-0.0491803,-0.555556,-0.884161,-0.171386,-0.805295,-0.766667,1
0,-0.155779,0.344262,-0.373737,-0.704492,0.138599,-0.867635,-0.933333,1
0,0.457286,0,0,0,0.317437,-0.528608,-0.666667,0
0,0.356784,0.114754,-0.151515,-0.408983,0.260805,-0.75491,-0.9,0
-0.882353,0.396985,0.0163934,-0.171717,0.134752,0.213115,-0.608881,0,1
0,0.738693,0.278689,-0.353535,-0.373522,0.385991,-0.0768574,0.233333,1
-0.529412,-0.00502513,0.180328,-0.656566,0,-0.23696,-0.815542,-0.766667,1
-0.0588235,0.949749,0.311475,0,0,-0.222057,-0.596072,0.533333,1
-0.764706,-0.165829,0.0655738,-0.434343,-0.843972,0.0968703,-0.529462,-0.9,1
-0.764706,-0.105528,0.47541,-0.393939,0,-0.00149028,-0.81725,-0.3,1
-0.529412,-0.00502513,0.114754,-0.232323,0,-0.0223547,-0.942784,-0.6,1
-0.529412,0.256281,0.147541,-0.636364,-0.711584,-0.138599,-0.089667,-0.2,0
-0.647059,-0.19598,0,0,0,0,-0.918019,-0.966667,1
-0.294118,0.668342,0.213115,0,0,-0.207153,-0.807003,0.5,1
-0.411765,0.105528,0.114754,0,0,-0.225037,-0.81725,-0.7,1
-0.764706,-0.18593,0.180328,-0.69697,-0.820331,-0.102832,-0.599488,-0.866667,1
-0.176471,0.959799,0.147541,-0.333333,-0.65721,-0.251863,-0.927412,0.133333,0
-0.294118,0.547739,0.213115,-0.353535,-0.543735,-0.126677,-0.350128,-0.4,1
-0.764706,0.175879,0.47541,-0.616162,-0.832151,-0.248882,-0.799317,0,1
-0.647059,-0.155779,0.180328,-0.353535,0,0.108793,-0.838599,-0.766667,1
-0.294118,0,0.114754,-0.171717,0,0.162444,-0.445773,-0.333333,0
-0.176471,-0.0552764,0.0491803,-0.494949,-0.813239,-0.00745157,-0.436379,-0.333333,1
-0.647059,-0.0351759,0.278689,-0.212121,0,0.111773,-0.863365,-0.366667,1
0.176471,-0.246231,0.344262,0,0,-0.00745157,-0.842015,-0.433333,1
0,0.809045,0.47541,-0.474747,-0.787234,0.0879285,-0.798463,-0.533333,0
-0.882353,0.306533,-0.0163934,-0.535354,-0.598109,-0.147541,-0.475662,0,1
-0.764706,-0.155779,-0.180328,-0.535354,-0.820331,-0.0938897,-0.239966,0,1
-0.0588235,0.20603,0.278689,0,0,-0.254843,-0.717336,0.433333,1
0.411765,-0.155779,0.180328,-0.373737,0,-0.114754,-0.81298,-0.166667,0
0,0.396985,0.0163934,-0.656566,-0.503546,-0.341282,-0.889838,0,1
0.0588235,-0.0854271,0.114754,0,0,-0.278688,-0.895816,0.233333,1
-0.764706,-0.0854271,0.0163934,0,0,-0.186289,-0.618275,-0.966667,1
-0.647059,-0.00502513,-0.114754,-0.616162,-0.79669,-0.23696,-0.935098,-0.9,1
-0.647059,0.638191,0.147541,-0.636364,-0.751773,-0.0581222,-0.837746,-0.766667,0
0.0588235,0.457286,0.442623,-0.313131,-0.609929,-0.0968703,-0.408198,0.0666667,0
0.529412,-0.236181,-0.0163934,0,0,-0.0223547,-0.912895,-0.333333,1
-0.294118,0.296482,0.47541,-0.858586,-0.229314,-0.415797,-0.569599,0.3,1
-0.764706,-0.316583,0.147541,-0.353535,-0.843972,-0.254843,-0.906917,-0.866667,1
-0.647059,0.246231,0.311475,-0.333333,-0.692671,-0.0104321,-0.806149,-0.833333,1
-0.294118,0.145729,0,0,0,0,-0.905209,-0.833333,1
0.0588235,0.306533,0.147541,0,0,0.0193741,-0.509821,-0.2,0
-0.647059,0.256281,-0.0491803,0,0,-0.0581222,-0.93766,-0.9,1
-0.647059,-0.125628,-0.0163934,-0.636364,0,-0.350224,-0.687447,0,1
-0.882353,-0.0251256,0.0491803,-0.616162,-0.806147,-0.457526,-0.811272,0,1
-0.647059,0.165829,0.213115,-0.69697,-0.751773,-0.216095,-0.975235,-0.9,1
0,0.175879,0.0819672,-0.373737,-0.555556,-0.0819672,-0.645602,-0.966667,1
0,0.115578,0.0655738,0,0,-0.266766,-0.502989,-0.666667,1
-0.764706,0.226131,-0.0163934,-0.636364,-0.749409,-0.111773,-0.454313,-0.966667,1
0,0.0753769,0.245902,0,0,0.350224,-0.480786,-0.9,1
-0.882353,-0.135678,0.0819672,0.0505051,-0.846336,0.230999,-0.283518,-0.733333,1
-0.294118,-0.0854271,0,0,0,-0.111773,-0.63877,-0.666667,1
-0.882353,-0.226131,-0.0819672,-0.393939,-0.867612,-0.00745157,0.00170794,-0.9,1
-0.529412,0.326633,0,0,0,-0.019374,-0.808711,-0.933333,0
0,0.0552764,0.47541,0,0,-0.117735,-0.898377,-0.166667,1
0,-0.427136,-0.0163934,0,0,-0.353204,-0.438941,0.533333,1
0,0.276382,0.311475,-0.252525,-0.503546,0.0819672,-0.380017,-0.933333,1
-0.647059,0.296482,0.508197,-0.010101,-0.63357,0.0849479,-0.239966,-0.633333,0
-0.0588235,0.00502513,0.213115,-0.191919,-0.491726,0.174367,-0.502135,-0.266667,0
-0.647059,0.286432,0.180328,-0.494949,-0.550827,-0.0342771,-0.59778,-0.8,0
0.176471,-0.0954774,0.393443,-0.353535,0,0.0402385,-0.362084,0.166667,0
-0.529412,-0.155779,0.47541,-0.535354,-0.867612,0.177347,-0.930828,-0.866667,1
-0.882353,-0.115578,0.278689,-0.414141,-0.820331,-0.0461997,-0.75491,-0.733333,1
-0.0588235,0.869347,0.47541,-0.292929,-0.468085,0.028316,-0.70538,-0.466667,0
-0.411765,0.879397,0.245902,-0.454545,-0.510638,0.299553,-0.183604,0.0666667,0
-0.529412,0.316583,0.114754,-0.575758,-0.607565,-0.0134128,-0.929974,-0.766667,1
-0.882353,0.648241,0.344262,-0.131313,-0.841608,-0.0223547,-0.775406,-0.0333333,1
-0.529412,0.899497,0.803279,-0.373737,0,-0.150522,-0.485909,-0.466667,1
-0.882353,0.165829,0.147541,-0.434343,0,-0.183308,-0.8924,0,1
-0.647059,-0.155779,0.114754,-0.393939,-0.749409,-0.0491803,-0.561913,-0.866667,1
-0.294118,0.145729,0.442623,0,0,-0.171386,-0.855679,0.5,1
-0.882353,-0.115578,0.0163934,-0.515152,-0.895981,-0.108793,-0.706234,-0.933333,1
-0.882353,-0.155779,0.0491803,-0.535354,-0.728132,0.0998511,-0.664389,-0.766667,1
-0.176471,0.246231,0.147541,-0.333333,-0.491726,-0.23994,-0.92912,-0.466667,1
-0.882353,-0.0251256,0.147541,-0.191919,0,0.135618,-0.880444,-0.7,1
-0.0588235,0.105528,0.245902,0,0,-0.171386,-0.864219,0.233333,1
0.294118,0.0351759,0.114754,-0.191919,0,0.377049,-0.959009,-0.3,1
0.294118,-0.145729,0.213115,0,0,-0.102832,-0.810418,-0.533333,1
-0.294118,0.256281,0.245902,0,0,0.00745157,-0.963279,0.1,0
0,0.98995,0.0819672,-0.353535,-0.352246,0.230999,-0.637916,-0.766667,0
-0.882353,-0.125628,0.114754,-0.313131,-0.817967,0.120715,-0.724167,-0.9,1
-0.294118,-0.00502513,-0.0163934,-0.616162,-0.87234,-0.198212,-0.642186,-0.633333,1
0,-0.0854271,0.311475,0,0,-0.0342771,-0.553373,-0.8,1
-0.764706,-0.0452261,-0.114754,-0.717172,-0.791962,-0.222057,-0.427839,-0.966667,1
-0.882353,-0.00502513,0.180328,-0.393939,-0.957447,0.150522,-0.714774,0,1
-0.294118,-0.0753769,0.0163934,-0.353535,-0.702128,-0.0461997,-0.994022,-0.166667,1
-0.529412,0.547739,0.180328,-0.414141,-0.702128,-0.0670641,-0.777968,-0.466667,1
0,0.21608,0.0819672,-0.393939,-0.609929,0.0223547,-0.893254,-0.6,0
-0.647059,-0.21608,0.147541,0,0,-0.0312965,-0.836038,-0.4,1
-0.764706,0.306533,0.57377,0,0,-0.326379,-0.837746,0,1
-0.647059,0.115578,-0.0491803,-0.373737,-0.895981,-0.120715,-0.699402,-0.966667,1
-0.764706,-0.0150754,-0.0163934,-0.656566,-0.716312,0.0342773,-0.897523,-0.966667,1
-0.882353,0.437186,0.409836,-0.393939,-0.219858,-0.102832,-0.304868,-0.933333,1
-0.882353,0.19598,-0.278689,-0.0505051,-0.851064,0.0581222,-0.827498,-0.866667,1
-0.294118,0.0854271,-0.278689,-0.59596,-0.692671,-0.28465,-0.372331,-0.533333,1
-0.764706,0.18593,0.311475,0,0,0.278689,-0.474808,0,0
0.176471,0.336683,0.114754,0,0,-0.195231,-0.857387,-0.5,1
-0.764706,0.979899,0.147541,1,0,0.0342773,-0.575576,0.366667,0
0,0.517588,0.47541,-0.0707071,0,0.254843,-0.749787,0,0
-0.294118,0.0954774,-0.0163934,-0.454545,0,-0.254843,-0.890692,-0.8,1
0.411765,0.21608,0.278689,-0.656566,0,-0.210134,-0.845431,0.366667,1
-0.0588235,0.00502513,0.245902,0,0,0.153502,-0.904355,-0.3,1
-0.0588235,0.246231,0.245902,-0.515152,0.41844,-0.14456,-0.479932,0.0333333,0
-0.882353,-0.0653266,-0.0819672,-0.777778,0,-0.329359,-0.710504,-0.966667,1
-0.0588235,0.437186,0.0819672,0,0,0.0402385,-0.956447,-0.333333,0
-0.294118,0.0351759,0.0819672,0,0,-0.275708,-0.853971,-0.733333,1
-0.647059,0.768844,0.409836,-0.454545,-0.631206,-0.00745157,-0.0811272,0.0333333,0
0,-0.266332,0,0,0,-0.371088,-0.774552,-0.866667,1
0.294118,0.115578,0.377049,-0.191919,0,0.394933,-0.276687,-0.2,0
-0.764706,0.125628,0.278689,0.010101,-0.669031,0.174367,-0.917165,-0.9,1
-0.647059,0.326633,0.311475,0,0,0.0253354,-0.723313,-0.233333,0
-0.764706,-0.175879,-0.147541,-0.555556,-0.728132,-0.150522,0.384287,-0.866667,1
-0.294118,0.236181,0.180328,-0.0909091,-0.456265,0.00149028,-0.440649,-0.566667,1
0,0.889447,0.344262,-0.717172,-0.562648,-0.0461997,-0.484202,-0.966667,0
0,-0.326633,0.245902,0,0,0.350224,-0.900939,-0.166667,1
-0.882353,-0.105528,-0.606557,-0.616162,-0.940898,-0.171386,-0.58924,0,1
-0.882353,0.738693,0.213115,0,0,0.0968703,-0.99146,-0.433333,0
-0.882353,0.0954774,-0.377049,-0.636364,-0.716312,-0.311475,-0.719044,-0.833333,1
-0.882353,0.0854271,0.442623,-0.616162,0,-0.19225,-0.725021,-0.9,1
-0.294118,-0.0351759,0,0,0,-0.293592,-0.904355,-0.766667,1
-0.882353,0.246231,0.213115,-0.272727,0,-0.171386,-0.981213,-0.7,1
-0.176471,0.507538,0.278689,-0.414141,-0.702128,0.0491804,-0.475662,0.1,0
-0.529412,0.839196,0,0,0,-0.153502,-0.885568,-0.5,0
-0.882353,0.246231,-0.0163934,-0.353535,0,0.0670641,-0.627669,0,1
-0.882353,0.819095,0.278689,-0.151515,-0.307329,0.19225,0.00768574,-0.966667,0
-0.882353,-0.0753769,0.0163934,-0.494949,-0.903073,-0.418778,-0.654996,-0.866667,1
0,0.527638,0.344262,-0.212121,-0.356974,0.23696,-0.836038,-0.8,1
-0.882353,0.115578,0.0163934,-0.737374,-0.56974,-0.28465,-0.948762,-0.933333,1
-0.647059,0.0653266,-0.114754,-0.575758,-0.626478,-0.0789866,-0.81725,-0.9,1
-0.647059,0.748744,-0.0491803,-0.555556,-0.541371,-0.019374,-0.560205,-0.5,0
-0.176471,0.688442,0.442623,-0.151515,-0.241135,0.138599,-0.394535,-0.366667,0
-0.294118,0.0552764,0.311475,-0.434343,0,-0.0312965,-0.316823,-0.833333,1
0.294118,0.386935,0.213115,-0.474747,-0.659574,0.0760059,-0.590948,-0.0333333,0
-0.647059,0.0653266,0.180328,0,0,-0.230999,-0.889838,-0.8,1
-0.294118,0.175879,0.57377,0,0,-0.14456,-0.932536,-0.7,1
-0.764706,-0.316583,0.0163934,-0.737374,-0.964539,-0.400894,-0.847139,-0.933333,1
0.0588235,0.125628,0.344262,-0.515152,0,-0.159463,0.028181,-0.0333333,0
0,0.19598,0,0,0,-0.0342771,-0.9462,-0.9,0
-0.764706,0.125628,0.409836,-0.151515,-0.621749,0.14456,-0.856533,-0.766667,1
-0.764706,-0.0753769,0.245902,-0.59596,0,-0.278688,0.383433,-0.766667,1
-0.294118,0.839196,0.540984,0,0,0.216095,0.181042,-0.2,1
0,-0.0552764,0.147541,-0.454545,-0.728132,0.296572,-0.770282,0,1
-0.764706,0.0854271,0.0491803,0,0,-0.0819672,-0.931682,0,1
-0.529412,-0.0954774,0.442623,-0.0505051,-0.87234,0.123696,-0.757472,-0.733333,1
0,0.256281,0.114754,0,0,-0.263785,-0.890692,0,1
0,0.326633,0.278689,0,0,-0.0342771,-0.730999,0,1
-0.411765,0.286432,0.311475,0,0,0.0312965,-0.943638,-0.2,1
-0.529412,-0.0552764,0.0655738,-0.555556,0,-0.263785,-0.940222,0,1
-0.176471,0.145729,0.0491803,0,0,-0.183308,-0.441503,-0.566667,0
0,0.0251256,0.278689,-0.191919,-0.787234,0.028316,-0.863365,-0.9,1
-0.764706,0.115578,-0.0163934,0,0,-0.219076,-0.773698,-0.933333,1
-0.882353,0.286432,0.344262,-0.656566,-0.567376,-0.180328,-0.968403,-0.966667,1
0.176471,-0.0753769,0.0163934,0,0,-0.228018,-0.923997,-0.666667,1
0.529412,0.0452261,0.180328,0,0,-0.0700447,-0.669513,-0.433333,0
-0.411765,0.0452261,0.213115,0,0,-0.14158,-0.935952,-0.1,1
-0.764706,-0.0552764,0.245902,-0.636364,-0.843972,-0.0581222,-0.512383,-0.933333,1
-0.176471,-0.0251256,0.245902,-0.353535,-0.78487,0.219076,-0.322801,-0.633333,0
-0.882353,0.00502513,0.213115,-0.757576,-0.891253,-0.418778,-0.939368,-0.766667,1
0,0.0251256,0.409836,-0.656566,-0.751773,-0.126677,-0.4731,-0.8,1
-0.529412,0.286432,0.147541,0,0,0.0223547,-0.807857,-0.9,1
-0.294118,0.477387,0.311475,0,0,-0.120715,-0.914603,-0.0333333,0
-0.529412,-0.0954774,0,0,0,-0.165425,-0.545687,-0.666667,1
-0.647059,0.0351759,0.180328,-0.393939,-0.640662,-0.177347,-0.443211,-0.8,1
-0.764706,0.577889,0.213115,-0.292929,0.0401891,0.174367,-0.952178,-0.7,1
-0.882353,0.678392,0.213115,-0.656566,-0.659574,-0.302534,-0.684885,-0.6,0
0,0.798995,-0.180328,-0.272727,-0.624113,0.126677,-0.678053,-0.966667,0
0.294118,0.366834,0.377049,-0.292929,-0.692671,-0.156483,-0.844577,-0.3,0
0,0.0753769,-0.0163934,-0.494949,0,-0.213115,-0.953032,-0.933333,1
-0.882353,-0.0854271,-0.114754,-0.494949,-0.763593,-0.248882,-0.866781,-0.933333,1
-0.882353,0.175879,-0.0163934,-0.535354,-0.749409,0.00745157,-0.668659,-0.8,1
-0.411765,0.236181,0.213115,-0.191919,-0.817967,0.0163934,-0.836892,-0.766667,1
-0.764706,0.20603,-0.114754,0,0,-0.201192,-0.678053,-0.8,1
-0.882353,0.0653266,0.147541,-0.434343,-0.680851,0.0193741,-0.945346,-0.966667,1
-0.764706,0.557789,-0.147541,-0.454545,0.276596,0.153502,-0.861657,-0.866667,0
-0.764706,0.0150754,-0.0491803,-0.292929,-0.787234,-0.350224,-0.934244,-0.966667,1
-0.882353,0.20603,0.311475,-0.030303,-0.527187,0.159464,-0.0742955,-0.333333,1
-0.647059,-0.19598,0.344262,-0.373737,-0.834515,0.0193741,0.0367208,-0.8,0
0.176471,0.628141,0.377049,0,0,-0.174367,-0.911187,0.1,1
-0.882353,1,0.245902,-0.131313,0,0.278689,0.123826,-0.966667,0
-0.0588235,0.678392,0.737705,-0.0707071,-0.453901,0.120715,-0.925705,-0.266667,0
0.0588235,0.457286,0.311475,-0.0707071,-0.692671,0.129657,-0.52263,-0.366667,0
-0.294118,0.155779,-0.0163934,-0.212121,0,0.004471,-0.857387,-0.366667,0
-0.882353,0.125628,0.311475,-0.0909091,-0.687943,0.0372578,-0.881298,-0.9,1
-0.529412,0.457286,0.344262,-0.636364,0,-0.0312965,-0.865927,0.633333,0
0.176471,0.115578,0.147541,-0.454545,0,-0.180328,-0.9462,-0.366667,0
-0.294118,-0.0150754,-0.0491803,-0.333333,-0.550827,0.0134128,-0.699402,-0.266667,1
0.0588235,0.547739,0.278689,-0.393939,-0.763593,-0.0789866,-0.926558,-0.2,1
-0.294118,0.658291,0.114754,-0.474747,-0.602837,0.00149028,-0.527754,-0.0666667,1
-0.882353,-0.00502513,-0.0491803,-0.79798,0,-0.242921,-0.596072,0,1
0.176471,-0.316583,0.737705,-0.535354,-0.884161,0.0581222,-0.823228,-0.133333,1
-0.647059,0.236181,0.639344,-0.292929,-0.432624,0.707899,-0.315115,-0.966667,1
-0.0588235,-0.0854271,0.344262,0,0,0.0611028,-0.565329,0.566667,1
-0.294118,0.959799,0.147541,0,0,-0.0789866,-0.786507,-0.666667,0
0.0588235,0.567839,0.409836,0,0,-0.260805,-0.870196,0.0666667,0
0,-0.0653266,-0.0163934,0,0,0.052161,-0.842015,-0.866667,1
-0.647059,0.21608,-0.147541,0,0,0.0730254,-0.958155,-0.866667,0
-0.764706,0.0150754,-0.0491803,-0.656566,-0.373522,-0.278688,-0.542272,-0.933333,1
-0.764706,-0.437186,-0.0819672,-0.434343,-0.893617,-0.278688,-0.783091,-0.966667,1
0,0.628141,0.245902,-0.272727,0,0.47839,-0.755764,-0.833333,0
0,-0.0452261,0.0491803,-0.212121,-0.751773,0.329359,-0.754056,-0.966667,1
-0.529412,0.256281,0.311475,0,0,-0.0372578,-0.608881,-0.8,0
-0.411765,0.366834,0.344262,0,0,0,-0.520068,0.6,1
-0.764706,0.296482,0.213115,-0.474747,-0.515366,-0.0104321,-0.561913,-0.866667,1
-0.647059,0.306533,0.0491803,0,0,-0.311475,-0.798463,-0.966667,1
-0.882353,0.0753769,-0.180328,-0.616162,0,-0.156483,-0.912041,-0.733333,1
-0.882353,0.407035,0.213115,-0.474747,-0.574468,-0.281669,-0.359522,-0.933333,1
-0.882353,0.447236,0.344262,-0.0707071,-0.574468,0.374069,-0.780529,-0.166667,0
-0.0588235,0.0753769,0.311475,0,0,-0.266766,-0.335611,-0.566667,1
0.529412,0.58794,0.868852,0,0,0.260805,-0.847139,-0.233333,0
-0.764706,0.21608,0.147541,-0.353535,-0.775414,0.165425,-0.309991,-0.933333,1
-0.176471,0.296482,0.114754,-0.010101,-0.704492,0.147541,-0.691716,-0.266667,0
-0.764706,-0.0954774,-0.0163934,0,0,-0.299553,-0.903501,-0.866667,1
-0.176471,0.427136,0.47541,-0.515152,0.134752,-0.0938897,-0.957301,-0.266667,0
-0.647059,0.698492,0.213115,-0.616162,-0.704492,-0.108793,-0.837746,-0.666667,0
0,-0.00502513,0,0,0,-0.254843,-0.850555,-0.966667,1
-0.529412,0.276382,0.442623,-0.777778,-0.63357,0.028316,-0.555935,-0.766667,1
-0.529412,0.18593,0.147541,0,0,0.326379,-0.29462,-0.833333,1
-0.764706,0.226131,0.245902,-0.454545,-0.527187,0.0700448,-0.654142,-0.833333,1
-0.294118,0.256281,0.278689,-0.373737,0,-0.177347,-0.584116,-0.0666667,0
-0.882353,0.688442,0.442623,-0.414141,0,0.0432191,-0.293766,0.0333333,0
-0.764706,0.296482,0,0,0,0.147541,-0.807003,-0.333333,1
-0.529412,0.105528,0.245902,-0.59596,-0.763593,-0.153502,-0.965841,-0.8,1
-0.294118,-0.19598,0.311475,-0.272727,0,0.186289,-0.915457,-0.766667,1
0.176471,0.155779,0,0,0,0,-0.843723,-0.7,0
-0.764706,0.276382,-0.245902,-0.575758,-0.208038,0.0253354,-0.916311,-0.966667,1
0.0588235,0.648241,0.278689,0,0,-0.0223547,-0.940222,-0.2,0
-0.764706,-0.0653266,0.0491803,-0.353535,-0.621749,0.132638,-0.491033,-0.933333,0
-0.647059,0.58794,0.0491803,-0.737374,-0.0851064,-0.0700447,-0.814688,-0.9,1
-0.411765,0.266332,0.278689,-0.454545,-0.947991,-0.117735,-0.691716,-0.366667,1
0.176471,0.296482,0.0163934,-0.272727,0,0.228018,-0.690009,-0.433333,0
0,0.346734,-0.0491803,-0.59596,-0.312057,-0.213115,-0.766012,0,1
-0.647059,0.0251256,0.213115,0,0,-0.120715,-0.963279,-0.633333,1
-0.176471,0.879397,-0.180328,-0.333333,-0.0732861,0.0104323,-0.36123,-0.566667,0
-0.647059,0.738693,0.278689,-0.212121,-0.562648,0.00745157,-0.238258,-0.666667,0
0.176471,-0.0552764,0.180328,-0.636364,0,-0.311475,-0.558497,0.166667,1
-0.882353,0.0854271,-0.0163934,-0.0707071,-0.579196,0.0581222,-0.712212,-0.9,1
-0.411765,-0.0251256,0.245902,-0.454545,0,0.0611028,-0.743809,0.0333333,0
-0.529412,-0.165829,0.409836,-0.616162,0,-0.126677,-0.795901,-0.566667,1
-0.882353,0.145729,0.0819672,-0.272727,-0.527187,0.135618,-0.819812,0,1
-0.882353,0.497487,0.114754,-0.414141,-0.699764,-0.126677,-0.768574,-0.3,0
-0.411765,0.175879,0.409836,-0.393939,-0.751773,0.165425,-0.852263,-0.3,1
-0.882353,0.115578,0.540984,0,0,-0.0223547,-0.840307,-0.2,1
-0.529412,0.125628,0.278689,-0.191919,0,0.174367,-0.865073,-0.433333,1
-0.882353,0.165829,0.278689,-0.414141,-0.574468,0.0760059,-0.64304,-0.866667,1
0,0.417085,0.377049,-0.474747,0,-0.0342771,-0.69684,-0.966667,1
-0.764706,0.758794,0.442623,0,0,-0.317437,-0.788215,-0.966667,1
-0.764706,-0.0753769,-0.147541,0,0,-0.102832,-0.9462,-0.966667,1
-0.647059,0.306533,0.278689,-0.535354,-0.813239,-0.153502,-0.790777,-0.566667,0
-0.0588235,0.20603,0.409836,0,0,-0.153502,-0.845431,-0.966667,0
-0.764706,0.748744,0.442623,-0.252525,-0.716312,0.326379,-0.514944,-0.9,0
-0.764706,0.0653266,-0.0819672,-0.454545,-0.609929,-0.135618,-0.702818,-0.966667,1
-0.764706,0.0552764,0.229508,0,0,-0.305514,-0.588386,0.0666667,1
-0.529412,-0.0452261,-0.0163934,-0.353535,0,0.0551417,-0.824082,-0.766667,1
0,0.266332,0.409836,-0.454545,-0.716312,-0.183308,-0.626815,0,1
-0.0588235,-0.346734,0.180328,-0.535354,0,-0.0461997,-0.554227,-0.3,1
-0.764706,-0.00502513,-0.0163934,-0.656566,-0.621749,0.0909091,-0.679761,0,1
-0.882353,0.0251256,0.213115,0,0,0.177347,-0.816396,-0.3,0
0.294118,0.20603,0.311475,-0.252525,-0.64539,0.260805,-0.396243,-0.1,0
-0.647059,0.0251256,-0.278689,-0.59596,-0.777778,-0.0819672,-0.725021,-0.833333,1
-0.882353,0.0954774,-0.0491803,-0.636364,-0.725768,-0.150522,-0.87959,-0.966667,1
0.0588235,0.407035,0.540984,0,0,-0.0253353,-0.439795,-0.2,0
0.529412,0.537688,0.442623,-0.252525,-0.669031,0.210134,-0.0640478,-0.4,1
0.411765,0.00502513,0.377049,-0.333333,-0.751773,-0.105812,-0.649872,-0.166667,1
-0.882353,0.477387,0.540984,-0.171717,0,0.469449,-0.760888,-0.8,0
-0.882353,-0.18593,0.213115,-0.171717,-0.865248,0.38003,-0.130658,-0.633333,1
-0.647059,0.879397,0.147541,-0.555556,-0.527187,0.0849479,-0.71819,-0.5,0
-0.294118,0.628141,0.0163934,0,0,-0.275708,-0.914603,-0.0333333,0
-0.529412,0.366834,0.147541,0,0,-0.0700447,-0.0572161,-0.966667,0
-0.882353,0.21608,0.278689,-0.212121,-0.825059,0.162444,-0.843723,-0.766667,1
-0.647059,0.0854271,0.0163934,-0.515152,0,-0.225037,-0.876174,-0.866667,1
0,0.819095,0.442623,-0.111111,0.205674,0.290611,-0.877028,-0.833333,0
-0.0588235,0.547739,0.278689,-0.353535,0,-0.0342771,-0.688301,-0.2,0
-0.882353,0.286432,0.442623,-0.212121,-0.739953,0.0879285,-0.163962,-0.466667,0
-0.176471,0.376884,0.47541,-0.171717,0,-0.0461997,-0.732707,-0.4,1
0,0.236181,0.180328,0,0,0.0819672,-0.846285,0.0333333,0
-0.882353,0.0653266,0.245902,0,0,0.117735,-0.898377,-0.833333,1
-0.294118,0.909548,0.508197,0,0,0.0581222,-0.829206,0.5,0
-0.764706,-0.115578,-0.0491803,-0.474747,-0.962175,-0.153502,-0.412468,-0.966667,1
0.0588235,0.708543,0.213115,-0.373737,0,0.311475,-0.722459,-0.266667,0
0.0588235,-0.105528,0.0163934,0,0,-0.329359,-0.945346,-0.6,1
0.176471,0.0150754,0.245902,-0.030303,-0.574468,-0.019374,-0.920581,0.4,1
-0.764706,0.226131,0.147541,-0.454545,0,0.0968703,-0.77626,-0.8,1
-0.411765,0.21608,0.180328,-0.535354,-0.735225,-0.219076,-0.857387,-0.7,1
-0.882353,0.266332,-0.0163934,0,0,-0.102832,-0.768574,-0.133333,0
-0.882353,-0.0653266,0.147541,-0.373737,0,-0.0938897,-0.797609,-0.933333,1
                                                                                                                                                                                                                                                                                                                                  exam.py                                                                                             000644  000766  000024  00000002342 13137603537 014265  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         import tensorflow as tf
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
print("test:", test)                                                                                                                                                                                                                                                                                              fromFile.py                                                                                         000644  000766  000024  00000002646 13137603537 015105  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         import tensorflow as tf
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
print("My Friends Scores will be \n", sess.run([hypothesis], feed_dict={X: [[50, 44, 78], [92, 88, 99]]}))                                                                                          ./._input_data 2.py                                                                                 000755  000766  000024  00000000337 13137603537 016164  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                             Mac OS X            	   2   �      �                                      ATTR       �   �   G                  �   G  com.apple.quarantine q/0083;596ddd77;The\x20Unarchiver;23BE86B9-FC00-410C-B59D-10E34AA66F0E                                                                                                                                                                                                                                                                                                  input_data 2.py                                                                                     000755  000766  000024  00000016662 13137603537 015622  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         # Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(work_directory):
    tf.gfile.MakeDirs(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with tf.gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with tf.gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels


class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.

    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()

  if fake_data:
    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
    data_sets.train = fake()
    data_sets.validation = fake()
    data_sets.test = fake()
    return data_sets

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000

  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)

  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)

  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)

  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)

  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
  data_sets.validation = DataSet(validation_images, validation_labels,
                                 dtype=dtype)
  data_sets.test = DataSet(test_images, test_labels, dtype=dtype)

  return data_sets

                                                                              input_data.py                                                                                       000644  000766  000024  00000016331 13137603537 015466  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         # Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(work_directory):
    tf.gfile.MakeDirs(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with tf.gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with tf.gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels


class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.

    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()

  if fake_data:
    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
    data_sets.train = fake()
    data_sets.validation = fake()
    data_sets.test = fake()
    return data_sets

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000

  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)

  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)

  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)

  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)

  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
  data_sets.validation = DataSet(validation_images, validation_labels,
                                 dtype=dtype)
  data_sets.test = DataSet(test_images, test_labels, dtype=dtype)

  return data_sets

                                                                                                                                                                                                                                                                                                       logisticRegression.py                                                                               000644  000766  000024  00000003030 13137603537 017204  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         import tensorflow as tf
import numpy as np


## 로지스틱 회귀 분석 (multi class )

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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        minimize.py                                                                                         000644  000766  000024  00000000773 13137603537 015162  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

hypothesis = X * W

## cost / loss

cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

for i in range(-30,50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

#show
plt.plot(W_val, cost_val)
plt.show()     placeholder.py                                                                                      000644  000766  000024  00000001741 13137603537 015617  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         import tensorflow as tf

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
                               queueFile.py                                                                                        000644  000766  000024  00000004242 13137603537 015260  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

# 다음 데이터는 익명의 학생들의 모의고사 3회점수와 기말고사 점수이다
## 모의 고사 3회의 점수와 기말고사 점수를 사용해 아직 시험전인 학생의 시험점수를 예측하라

## 파일 로드 ## 텐서에서 제공하는 파일 큐 - 여러개의 파일을 배치로 가져다 쓸수 있다.
filename_queue = tf.train.string_input_producer(['./data-01-test-score.csv', './data-02-test-score.csv'], shuffle=False, name='scoreCard')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

print(key)

print(value)
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

## optimizer - 텐서플로에서 제공하는 경사하강법 코스트를 최소화 하게 설정 학습 간격은 0.01
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
print("My Friends Scores will be \n", sess.run([hypothesis], feed_dict={X: [[50, 44, 78], [92, 88, 99]]}))                                                                                                                                                                                                                                                                                                                                                              test.py                                                                                             000644  000766  000024  00000001154 13137603537 014312  0                                                                                                    ustar 00hyungeunjung                    staff                           000000  000000                                                                                                                                                                         import tensorflow as tf

x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

## 공식
hypothesis = x_train * W + b

## Cost / Loss Function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))  ## 평균 reduce_mean

## optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(cost),sess.run(W),sess.run(b))


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    