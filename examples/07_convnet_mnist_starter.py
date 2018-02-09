""" Using convolutional net on MNIST dataset of handwritten digits
MNIST dataset: http://yann.lecun.com/exdb/mnist/
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 07
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import tensorflow as tf

import utils

def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    '''
    A method that does convolution + relu on inputs
    '''
    #############################
    ########## TO DO ############
    #############################
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        c_r = tf.layers.conv2d(
            inputs = inputs,
            filters = filters,
            kernel_size = k_size,
            strides = stride,
            padding = padding,
            activation = tf.nn.relu
        )
        return c_r

def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    '''A method that does max pooling on inputs'''
    #############################
    ########## TO DO ############
    #############################
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        mp = tf.layers.max_pooling2d(
            inputs = inputs,
            pool_size = ksize,
            strides = stride,
            padding = padding
        )
        return mp

def fully_connected(inputs, out_dim, scope_name='fc'):
    '''
    A fully connected linear layer on inputs
    '''
    #############################
    ########## TO DO ############
    #############################
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        fc = tf.layers.dense(
            inputs=inputs,
            units=out_dim,
            activation=tf.nn.relu
        )
        return fc

class ConvNet(object):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                trainable=False, name='global_step')
        self.n_classes = 10
        self.skip_step = 20
        self.n_test = 10000
        self.training = False

    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.get_mnist_dataset(self.batch_size)
            iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])
            # reshape the image to make it work with tf.nn.conv2d

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)    # initializer for train_data

    def inference(self):
        '''
        Build the model according to the description we've shown in class
        '''
        #############################
        ########## TO DO ############
        #############################
        # conv_relu(inputs, filters, k_size, stride, padding, scope_name)
        # maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool')
        # fully_connected(inputs, out_dim, scope_name='fc')

        conv1 = conv_relu(
            inputs = self.img,
            filters = 32,
            k_size = [5,5],
            stride = 1,
            padding = 'SAME',
            scope_name = 'conv1'
        )
        maxpool1 = maxpool(
            inputs = conv1,
            ksize = [2,2],
            stride = 2,
            scope_name = 'maxpool1'
        )
        conv2 = conv_relu(
            inputs = maxpool1,
            filters = 64,
            k_size = [5,5],
            stride = 1,
            padding = 'SAME',
            scope_name = 'conv2'
        )
        maxpool2 = maxpool(
            inputs = conv2,
            ksize = [2,2],
            stride = 2,
            scope_name = 'maxpool2'
        )
        # reshape maxpool2 to a 1-d vector?
        feature_dim = maxpool2.shape[1] * maxpool2.shape[2] * maxpool2.shape[3]
        maxpool2 = tf.reshape(maxpool2, [-1, feature_dim])
        fc1 = fully_connected(
            inputs = maxpool2,
            out_dim = 1024,
            scope_name = 'fc1'
        )
        dropout = tf.layers.dropout(fc1,
                                    self.keep_prob,
                                    training=self.training,
                                    name='dropout')
        self.logits = tf.layers.dense(dropout, self.n_classes, name='logits')

    def loss(self):
        '''
        define loss function
        use softmax cross entropy with logits as the loss function
        tf.nn.softmax_cross_entropy_with_logits
        softmax is applied internally
        don't forget to compute mean cross all sample in a batch
        '''
        #############################
        ########## TO DO ############
        #############################
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels = self.label,
                logits = self.logits,
                name = 'entropy'
            )
            self.loss = tf.reduce_mean(entropy, name = 'loss')

    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        Don't forget to use global step
        '''
        #############################
        ########## TO DO ############
        #############################
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                global_step=self.gstep)

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        Remember to track both training loss and test accuracy
        '''
        #############################
        ########## TO DO ############
        #############################
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/convnet_starter/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/convnet_starter')
        writer = tf.summary.FileWriter('./graphs/convnet_starter', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_starter/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()

if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=15)
