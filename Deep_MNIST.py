import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

# Create input object which reads data from MNIST datasets. 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# User Interactive session makes it the default session so we don't need to pass sess
sess = tf.InteractiveSession()

# Define placeholders for MNIST input data
x = tf.placeholder(tf.float32, shape=[None, 784]) # None because we dont't know how many images we're going to pass in, 784 because of image size (28 x 28 px)
y_ = tf.placeholder(tf.float32, [None, 10]) # None because we don't know how many images we're going to pass in, 10 because 0-9


# change the MNISt input data from a list of values to a 28 pixel x 28 pixel x 1 grayscale value cube which th eConvolution NN can use.
x_image =  tf.reshape(x, [-1,28,28,1], name="x_image");  # -1 is a flag that says 'make this a list of the other dementions'

# We are using RELU as our activation function
# RELU = if x < 0 then y= 0 else if x > 0 y = x


def weight_variable(shape):
    initial =  tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and Pooling - we do Convolution, then pooling to control overfitting
# 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Define layers in the NN
# 32 features for each 5x5 patch of the image
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Do convolution on images, add bias, and push through RELU activation function
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# Take result and run through max_pool
h_pool1 = max_pool_2x2(h_conv1)

# 2nd convolution layer
# Process the 32 features from convolution layer 1, in 5x5 patch. Return 64 features weights and balances
W_conv2 = weight_variable([5,5, 32, 64]) # 5x5 images, take 32 features, returns 64 features
b_conv2 = bias_variable([64])
# Do convolution of the output of the 1st convolution layer. Pool results.
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # Reduces the output to a 7x7 image

# Fully Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# Connect output of pooling layer 2 as input to fully conneted layer
h_pool2_flat = tf.reshape(h_pool2, [ -1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout some neurons to reduce overfitting ( Sets some weights and balances to 0 to make sure its not overtrained on training data)
keep_prob = tf.placeholder(tf.float32) # get dropout probability as a training input
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Define model as (value * weights) + bias
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss measurement defined as cross_entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# Loss optimization
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Accuracy calculation
# What is correct
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# How accurate is it?
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize all the variables
sess.run(tf.global_variables_initializer())

# Train the model
# Define the number of steps and how often we display progress
num_steps = 3000
display_every = 100

start_time = time.time()
end_time = time.time()

# Training loop
for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    if i % display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        end_time = time.time()
        print("Step {0}, elapsed time {1: .2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100.0))

# Display summary
#   Time to train
end_time = time.time()
print("Total training time for {0} batches: {1: .2f} seconds".format(1+1, end_time-start_time))

# Accuracy on test data
print("Test accuracy {0: 3f}%".format(accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100))

sess.close()
