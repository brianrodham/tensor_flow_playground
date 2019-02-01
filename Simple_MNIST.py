import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # Only return the highest probability digit

# x is placeholder for the 28 X 28 image data
x = tf.placeholder(tf.float32, shape=[None, 784]) # 784 pixels

# y_ is called "y bar" and is a 10 element vector, containing the predicted probability of each
# digit (0-9) class.
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Variables that will change as we train - Weights and balances
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define our model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Loss is cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Each training step in gradient decent we want to minimize cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # Set learning rate at 0.5

# Initialize the global variables
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

# Use 1000 of the MNIST images to train the model
for i in range(1000): 
    batch_xs, batch_ys = mnist.train.next_batch(100) # Get 100 random data points. batch_xs = image. batch_ys = actual digit (0-9).

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # do the optimization with this data

# Evaluate how well the model did. Do this by comparing the digit with the highest probability in actual y vs predicted y
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # Gives a vector of true or false depending on if it model guessed right
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # Takes the mean of the correct_prediction vector
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Test Accuracy: {0}%".format(test_accuracy * 100.0))

sess.close()