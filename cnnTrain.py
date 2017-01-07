import tensorflow as tf
import numpy as np
import glob
import re
import cv2
import cPickle as pickle
from sklearn import preprocessing

class dataSet:
    def __init__(self):
        self.WIDTH = None
        self.HEIGHT = None
        self.current_index = 0
        self.no_samples = 0

    def tokenize(self,filename):
        digits = re.compile(r'(\d+)')
        return tuple(int(token) if match else token
                for token, match in
                ((fragment, digits.search(fragment))
                for fragment in digits.split(filename)))

    def create_input_matrix(self,input_folder):
        self.input_files = []
        self.labels = []
        files = glob.glob(input_folder + "*")
        files.sort(key = self.tokenize)
        self.no_samples = len(files)
        print self.no_samples
        for i in xrange(self.no_samples):
            img = cv2.imread(files[i],0)
            img = cv2.resize(img,(112,80))
            self.input_files.append(img)
            if i/400 == 0:
                self.labels.append(np.array([1,0,0]))
            elif i/400 == 1:
                self.labels.append(np.array([0,1,0]))
            else:
                self.labels.append(np.array([0,0,1]))
        print img.shape
        self.HEIGHT, self.WIDTH = img.shape
        self.input_files = np.array(self.input_files)
        scaler = preprocessing.MinMaxScaler()
        self.vector_length = self.WIDTH*self.HEIGHT
        self.input_files = self.input_files.reshape(self.no_samples,self.vector_length)
        scaler.fit(self.input_files)
        self.input_files = scaler.transform(self.input_files)
        self.labels = np.array(self.labels)
        perm = np.arange(self.no_samples)
        np.random.shuffle(perm)
        self.input_files = self.input_files[perm]
        self.labels = self.labels[perm]

    def get_vector_length(self):
        return self.vector_length

    def get_batch(self, batch_size):
        if batch_size > self.no_samples: batch_size = self.no_samples
        start = self.current_index
        end = self.current_index + batch_size
        self.current_index += batch_size

        if self.current_index > self.no_samples:
            perm = np.arange(self.no_samples)
            np.random.shuffle(perm)
            self.input_files = self.input_files[perm]
            self.labels = self.labels[perm]
            start = 0
            end = batch_size
        
        return self.input_files[start:end], self.labels[start:end]


def weight_variable(var_string,shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)        #tf.get_variable(var_string, shape=shape,
           #initializer=tf.contrib.layers.xavier_initializer())#

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME') 

print "Reading input"

training_folder = "/Users/11162/GHC/ChecksVsStripesVsSolids/"
input_data = dataSet()
input_data.create_input_matrix(training_folder)
vector_length = input_data.get_vector_length()
print "Reading input done"

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, vector_length])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

W_conv1 = weight_variable("W_conv1",[5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,input_data.WIDTH,input_data.HEIGHT,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable("W_conv2",[5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable("W_fc1",[input_data.WIDTH/4 * input_data.HEIGHT/4 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, input_data.WIDTH/4 * input_data.HEIGHT/4 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable("W_fc2",[1024, 3])
b_fc2 = bias_variable([3])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

print "Done creating TF graph. Starting training"

for i in range(100):
    batch = input_data.get_batch(100)

    if i%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

pickle.dump(accuracy,open('cnn.p','w'))

