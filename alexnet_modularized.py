from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from datetime import datetime
import math
import time

from six.moves import xrange

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100,
                            """Number of batches to run.""")


def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def kernel_gen(shape):
    return tf.Variable(tf.truncated_normal(shape, dtype=tf.float32,
                                        stddev=1e-1), name='weights')


def conv_gen(conv_input, kernel, strides):
    return tf.nn.conv2d(conv_input, kernel, strides, padding='SAME')


def biases_gen(shape_size):
    return tf.Variable(tf.constant(0.0, shape=[shape_size], dtype=tf.float32),
                    trainable=True, name='biases')


def max_pool_gen(conv, pool_name):
    return tf.nn.max_pool(conv, ksize=[1,3,3,1],
                            strides=[1,2,2,1],
                            padding='VALID',
                            name=pool_name)


def conv_step(layer_name, conv_input, kernel_shape, conv_strides, biases_shape_num, parameters):
    with tf.name_scope(layer_name) as scope:
        kernel = kernel_gen(kernel_shape)
        conv = conv_gen(conv_input, kernel, conv_strides)
        biases = biases_gen(biases_shape_num)
        
        bias = tf.nn.bias_add(conv, biases)
        conv_activ = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    
    return conv_activ, parameters


def inference(images):
    parameters = []

    # 1st conv, 1st max pooling
    conv1, parameters = conv_step('conv1', images, [11,11,1,64], [1,1,1,1], 64, parameters) 
    print_activations(conv1)
    pool1 = max_pool_gen(conv1, 'pool1')
    print_activations(pool1)

    # 2nd conv, 2nd max pooling
    conv2, parameters = conv_step('conv2', pool1, [5,5,64,192], [1,1,1,1], 192, parameters)
    print_activations(conv2)
    pool2 = max_pool_gen(conv2, 'pool2')
    print_activations(pool2)

    # 3rd conv
    conv3, parameters = conv_step('conv3', pool2, [3,3,192,384], [1,1,1,1], 384, parameters)
    print_activations(conv3)

    # 4th conv
    conv4, parameters = conv_step('conv4', conv3, [3,3,384,256], [1,1,1,1], 256, parameters)
    print_activations(conv4)

    # 5th conv, 5th max pooling
    conv5, parameters = conv_step('conv5', conv4, [3,3,256,256], [1,1,1,1], 256, parameters)
    print_activations(conv5)
    pool5 = max_pool_gen(conv5, 'pool5')
    print_activations(pool5)
    print(pool5)
    return pool5, parameters



def time_tensorflow_run(session, target, info_string):
  """Run the computation to obtain the target tensor and print timing stats.

  Args:
    session: the TensorFlow session to run the computation under.
    target: the target Tensor that is passed to the session's run() function.
    info_string: a string summarizing this run, to be printed with the stats.

  Returns:
    None
  """
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  for i in xrange(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target)
    duration = time.time() - start_time
    if i > num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))



def run_benchmark():
  """Run the benchmark on AlexNet."""
  with tf.device('/gpu:0'): 
      with tf.Graph().as_default():
        '''
        # Generate some dummy images.
        image_size = 28 
        # Note that our padding definition is slightly different the cuda-convnet.
        # In order to force the model to start with the same activations sizes,
        # we add 3 to the image_size and employ VALID padding above.
        images = tf.Variable(tf.random_normal([FLAGS.batch_size,
                                               image_size,
                                               image_size, 3],
                                              dtype=tf.float32,
                                              stddev=1e-1))
         '''
        images = mnist

        # set input and target output classes
        x = tf.placeholder(tf.float32, shape=[None,28,28,1])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        # Build an initialization operation.
        #init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        #config = tf.ConfigProto()
        #config.gpu_options.allocator_type = 'BFC'
        #sess = tf.Session(config=config)
        sess = tf.InteractiveSession()
        #sess.run(init)


        # Build a Graph that computes the logits predictions from the
        # inference model.
        pool5, parameters = inference(x)
        

        # desnely connected layer
        W_fc1 = kernel_gen([2*2*256, 100])
        b_fc1 = biases_gen(100)

        h_pool_flat = tf.reshape(pool5, [-1, 2*2*256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
        print_activations(h_fc1)
        
        W_fc2 = kernel_gen([100, 100])
        b_fc2 = biases_gen(100)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        print_activations(h_fc2)
        
        # readout
        W_fc3 = kernel_gen([100, 10])
        b_fc3 = biases_gen(10)

        y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
         

        # training and testing
         
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.initialize_all_variables())
        for i in range(20000):
            batch = mnist.train.next_batch(50)   
            batch_imgs, batch_labs = batch
            #print(batch_imgs.shape)
            #print(np.reshape(batch_imgs, (128,28,28,1)).shape)

            # PROTIP: do preproce before loop
            batch_imgs = np.reshape(batch_imgs, (50,28,28,1))
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch_imgs, y_: batch_labs})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            train_step.run(feed_dict={x: batch_imgs, y_: batch_labs})
        
        mnist_test_imgs_reshaped = np.reshape(mnist.test.images, (10000,28,28,1))

        print("test acuracy %g" % accuracy.eval(feed_dict={
            x: mnist_test_imgs_reshaped, y_: mnist.test.labels}))

        ''' 
        # Run the forward benchmark.
        time_tensorflow_run(sess, pool5, "Forward")

        # Add a simple objective so we can calculate the backward pass.
        objective = tf.nn.l2_loss(pool5)
        # Compute the gradient with respect to all the parameters.
        grad = tf.gradients(objective, parameters)
        # Run the backward benchmark.
        time_tensorflow_run(sess, grad, "Forward-backward")
        '''


def main(_):
  run_benchmark()


if __name__ == '__main__':
  tf.app.run()

