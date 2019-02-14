import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layer
from absl import flags
import random
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

class zsModel:
    
    def __init__(self, hyperparam, name):
        
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
        
        self.build_network()
            
    def build_network(self):
        
        self.graph = tf.Graph()
        with self.graph.as_default():

            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.8
            self.sess = tf.Session(config=config)
            with tf.variable_scope(self.name):
                self.x = tf.placeholder(tf.float32, (self.input_shape))
                self.y = tf.placeholder(tf.float32, (self.output_shape))
            
                z = layer.conv2d(self.x, self.filters[0][0], self.filters[0][1])
                for filter in self.filters[1:]:
                    z = layer.conv2d(z, filter[0], kernel_size = filter[1], padding = filter[3])
                    if (filter[2]):
                        z = layer.max_pool2d(z, 2)
                        #z = tf.layers.batch_normalization(z)
                
                print(z.shape)
                z = layer.flatten(z)
                z = layer.fully_connected(z, 1024)
                
                #Load in embedding weights
                #embeddings = np.load("embedding_matrix.npy")
                #embedding_weights = tf.Variable(np.transpose(embeddings.astype(np.float32)), trainable = False)
                #z = layer.fully_connected(z, int(embeddings.shape[-1])) #Shape received normally as float
               
                #self.img_extract = z
                #Feature projection layer
                #extract = tf.matmul(z, embedding_weights)
                self.out = layer.fully_connected(z, self.output_shape[-1], activation_fn = None)
            
                vec_loss = tf.squared_difference(self.out, self.y)
                self.loss = tf.reduce_mean(vec_loss)
                #self.create_summaries()
                
                #Write a summary for loss
                #self.train_writer = tf.summary.FileWriter("{}-{}".format("tfboard", self.name), self.graph)
                
                self.train_op = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(self.loss)
                
                self.saver = tf.train.Saver()
                
                self.sess.run(tf.global_variables_initializer())
            
    def train(self, input_set, label_set, epochs):
        
        with self.graph.as_default():
        
            local_input = input_set
            local_output = label_set
            
            #Eventually returns array of epoch losses
            accumLoss = []
            batchIdx = self.batch_size
            last_batchIdx = 0
            step = 0
            
            for i in range(epochs):
                
                #epochLossArray = []
                accuracyStore = []
                
                print("{}th epoch".format(i))
                while (batchIdx < input_set.shape[0]):
                    batch_train_x = local_input[last_batchIdx:batchIdx]
                    batch_train_y = local_output[last_batchIdx:batchIdx]
                    
                    _, pred, stepLoss= self.sess.run([self.train_op, self.out, self.loss], feed_dict = {self.x: batch_train_x, self.y: batch_train_y})
                    print(stepLoss)
                    
                    accuracy = 0
                    accumLoss.append(stepLoss)
                    for i,v in enumerate(batch_train_y):
                        if np.equal(np.argmax(v), np.argmax(pred[i])):
                            accuracy += 1
                    accuracy /= len(batch_train_y)
                    accuracyStore.append(accuracy*100)
                    accuracy = 0
                    
                    #epochLossArray.append(stepLoss)
                    step += 1
                    if (batchIdx + self.batch_size <= input_set.shape[0]):
                        batchIdx += self.batch_size
                        last_batchIdx += self.batch_size
                    else:
                        last_batchIdx = batchIdx
                        batchIdx = input_set.shape[0]
                                
                    if (step%2000 == 0):
                        print("SAVING MODEL, {}th step".format(step))
                        print("Averaged Accuracy {}".format(np.sum(np.array(accuracyStore))/len(accuracyStore)))
                        
                        #Display Accuracy Progression per Epoch
                        plt.plot(accuracyStore)
                        plt.ylabel("Accuracy Percentile")
                        plt.xlabel("Epoch step")
                        plt.show()
                        accuracyStore = []
                        self.saver.save(self.sess, "models/newModel.ckpt")
                
                #Reset batches
                batchIdx = self.batch_size
                last_batchIdx = 0
                
                #Arbitrarily shuffle data
                
                local_input, local_output = self._shuffle(input_set, label_set)
                
            #Return accumulated epoch loss
            return accumLoss
            
            
    
    def _shuffle(self, train_input, train_output):
        perm_idx = np.random.permutation(train_input.shape[0])
        new_train_in = []
        new_train_out = []
        
        for idx in perm_idx:
            new_train_in.append(train_input[idx])
            new_train_out.append(train_output[idx])
            
        return np.array(new_train_in), np.array(new_train_out)

    #def featurePredicton(self, input_set):
        #Returns prediction for the feature map on the second last layer
        
        #return self.sess.run(self.img_extract, feed_dict = {self.x: input_set})

    def create_summaries(self):
        with self.graph.as_default():
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()
        
    def load_weights(self, modelName):
        #Loads weights from previous save
        
        self.saver.restore(self.sess, "models/{}".format(modelName))
    
    def accuracy_test(self, data_x, data_y):
        accuracy = 0
        print(data_x.shape)
        for x, y in zip(data_x, data_y):
            x = np.expand_dims(x, axis = 0) #Ensure 4 dimensional input
            pred = self.sess.run(self.out, feed_dict = {self.x: x})
            if (np.equal(np.argmin(pred), np.argmin(data_y))):
                accuracy += 1
        accuracy /= data_x.shape[0]
        print("Final accuracy for test set: {}".format(accuracy))
