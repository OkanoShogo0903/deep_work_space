import sys
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print (mnist.train.num_examples)
print (mnist.test.num_examples)
print (mnist.validation.num_examples)

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

# $B@52r%G!<%?$rF~$l$kMQ$N(Bplaceholder
y = tf.placeholder(tf.float32, [None, 10])

# weight$B$rMQ0U(B
# $B3F%T%/%;%k$KBP$9$k(B0$B!A(B9$B$^$G$N?t;z$KBP$9$k(Bweight$B$rF~$l$k$H$$$&0UL#$G!"(B784*10$B8DMQ0U(B
w = tf.Variable(tf.zeros([784, 10]))

# $B<0!J(Bx * w$B!K(B
f = tf.matmul(x, w)

#print ("shape", f)
# $B%F%-%H!<$J(Bloss function$B$N$h$&$J%J%K%+(B
loss = tf.reduce_sum(tf.abs(y - f) * 0.1)

# $B0lHV$h$/8+$+$1$k(BGradientDescentOptimizer$B!J:G5^9_2<K!!K$r;H$&(B
opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# valiable$B$N=i4|2=(B
sess = tf.Session()
sess.run(tf.initialize_all_variables)

# $B3X=,3+;O(B
sess.run(opt, feed_dict={x: mnist.train.images, y: mnist.train.labels})

sys.exit() # <--
# $B7k2L(B
weights = sess.run(w)

# $B;;=P$5$l$?(Bweight$B$rIA2h$7$F$_$k(B
f, axarr = plt.subplots(2, 5)
for idx in range(10):
    ax = axarr[int(idx / 5)][idx % 5]
    ax.imshow(weights[:, idx].reshape(28, 28), cmap = cm.Greys_r)
    ax.set_title(str(idx))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
