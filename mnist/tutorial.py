import sys
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print (mnist.train.num_examples)
print (mnist.test.num_examples)
print (mnist.validation.num_examples)

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

# 正解データを入れる用のplaceholder
y = tf.placeholder(tf.float32, [None, 10])

# weightを用意
# 各ピクセルに対する0〜9までの数字に対するweightを入れるという意味で、784*10個用意
w = tf.Variable(tf.zeros([784, 10]))

# 式（x * w）
f = tf.matmul(x, w)

#print ("shape", f)
# テキトーなloss functionのようなナニカ
loss = tf.reduce_sum(tf.abs(y - f) * 0.1)

# 一番よく見かけるGradientDescentOptimizer（最急降下法）を使う
opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# valiableの初期化
sess = tf.Session()
sess.run(tf.initialize_all_variables)

# 学習開始
sess.run(opt, feed_dict={x: mnist.train.images, y: mnist.train.labels})

sys.exit() # <--
# 結果
weights = sess.run(w)

# 算出されたweightを描画してみる
f, axarr = plt.subplots(2, 5)
for idx in range(10):
    ax = axarr[int(idx / 5)][idx % 5]
    ax.imshow(weights[:, idx].reshape(28, 28), cmap = cm.Greys_r)
    ax.set_title(str(idx))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
