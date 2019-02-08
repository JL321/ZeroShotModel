import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layer
from absl import flags
import random

FLAGS = flags.FLAGS

class zsModel:
    
    def __init__(self, hyperparam, name):
        
        print(FLAGS.summaries_dir)
        self.name = name
        self.hparams = hyperparam
        self.filters = self.hparams.filters
        
        """
        filters is (n,3)
        n1 = size, n2 = kernel_size, n3 = use pooling
        """
        
        self.input_shape = self.hparams.input_shape
        self.feature_size = self.hparams.feature_shape
        self.output_shape = self.hparams.output_shape
        self.batch_size = self.hparams.batch_size 
        
    def build_graph(self):
        
        #Build the main graph
        self.x = tf.placeholder(tf.float32, (self.input_shape))
        self.y = tf.placeholder(tf.float32, (self.output_shape))
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            
            self.sess = tf.Session()
            self.sess.run(self._build_network())
            self.sess.run(tf.global_variables_initializer())
        
        
    def _build_network(self):
        
        with tf.variable_scope(self.name):
            z = layer.conv2d(self.x, self.filters[0][0], self.filters[0][1])
            for filter in self.filters[1:]:
                z = layer.conv2d(z, filter[0], kernel_size = filter[1])
                if (filter[2]):
                    z = layer.max_pool2d(z, 2)
            z = layer.flatten(z)
            z = layer.fully_connected(z, 4096)
            
            #Feature projection layer
            extract = layer.fully_connected(z, self.feature_size)
            out = layer.fully_connected(extract, self.output_shape, trainable = False)
        
            self.output = out
            self.featureExtract = extract
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.y, logits = self.output)
            tf.summary.scalar("Loss", self.loss)
            merged = tf.summary.merge_all()
            
            #Write a summary for loss
            train_writer = tf.summary.FileWriter("{}-{}".format(FLAGS.summaries_dir, self.name), self.graph)
            
            self.train = tf.train.AdamOptimizer.minimize(self.loss)
    
    
        train_writer.add_summary(merged)
    
    def train(self, input_set, label_set, epochs):
        
        #Eventually returns array of epoch losses
        epochLoss = 0
        accumLoss = []
        
        batchIdx = self.batch_size
        last_batchIdx = 0
        step = 0
        
        saver = tf.train.Saver()
        
        for _ in range(epochs):
            
            while (batchIdx < input_set.shape[0]):
                batch_train_x = input_set[last_batchIdx:batchIdx]
                batch_train_y = label_set[last_batchIdx:batchIdx]
                stepLoss = self.sess.run([self.train, self.loss], feed_dict = {self.x: batch_train_x, self.y: batch_train_y})
                epochLoss += stepLoss
                step += 1
                if (batchIdx + self.batch_size <= input_set.shape[0]):
                    batchIdx += self.batch_size
                    last_batchIdx += self.batch_size
                else:
                    last_batchIdx = batchIdx
                    batchIdx = input_set.shape[0]
            
            if (step%5000 == 0):
                saver.save(self.sess, "/models/model.ckpt")
            accumLoss.append(epochLoss)
            
            #Reset batches
            batchIdx = self.batch_size
            last_batchIdx = 0
            epochLoss = 0
            
            #Arbitrarily shuffle data
            input_set, label_set = self._shuffle(input_set, label_set)
                
            
        #Return accumulated epoch loss
        return accumLoss
    
    def _shuffle(self, train_input, train_output):
        combined = list(zip(train_input, train_output))
        random.shuffle(combined)
        
        shuffled_x, shuffled_y = zip(*combined)
        return shuffled_x, shuffled_y

    def featurePredicton(self, input_set):
        #Returns prediction for the feature map on the second last layer
        
        return self.sess.run(self.featureExtract, feed_dict = {self.x: input_set})

    
        
        
        
    