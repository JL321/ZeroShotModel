import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layer
from absl import flags
import random
import matplotlib.pyplot as plt

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
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            self.sess = tf.Session(config=config)
            with tf.variable_scope(self.name):
                self.x = tf.placeholder(tf.float32, (self.input_shape))
                self.y = tf.placeholder(tf.float32, (self.output_shape))
            
                z = layer.conv2d(self.x, self.filters[0][0], self.filters[0][1])
                for filtera in self.filters[1:]:
                    z = layer.conv2d(z, filtera[0], kernel_size = filtera[1], padding = filtera[3])
                    if (filtera[2]):
                        z = layer.max_pool2d(z, 2)
                    if (filtera[3]):
                        z = tf.layers.batch_normalization(z)
                        z = tf.layers.dropout(z, .8)
                
                print(z.shape)
                z = layer.flatten(z)
                z = layer.fully_connected(z, 1024)
                z = tf.layers.batch_normalization(z)
                #Load in embedding weights
                regularizer = tf.contrib.layers.l2_regularizer(scale = 0.2)
                
                embeddings = np.load("embedding_matrix.npy")
                embedding_weights = tf.Variable(np.transpose(embeddings.astype(np.float32)), trainable = False)
                extract = layer.fully_connected(z, int(embeddings.shape[-1]), weights_regularizer = regularizer) #Shape received normally as float
                 
                self.img_extract = extract
                extract = tf.layers.batch_normalization(extract)
                #Feature projection layer
                
                self.out = tf.matmul(extract, embedding_weights)
                        
                #vec_loss = tf.squared_difference(self.out, self.y)
                vec_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.y, logits = self.out)
                self.loss = tf.reduce_mean(vec_loss) + tf.losses.get_regularization_loss()
                #self.create_summaries()
                
                #Write a summary for loss
                #self.train_writer = tf.summary.FileWriter("{}-{}".format("tfboard", self.name), self.graph)
                
                self.train_op = tf.train.GradientDescentOptimizer(learning_rate = 0.002).minimize(self.loss)
                
                self.saver = tf.train.Saver()
                
                self.sess.run(tf.global_variables_initializer())
            
    def train(self, local_x, local_y, steps):
        
        with self.graph.as_default():
 
            #Eventually returns array of epoch losses
            accumLoss = []
            accuracyStore = []
            #done_epoch = False
            
            #epochLossArray = []
            #while (done_epoch == False):
            for step in range(steps):
                                
                tempStore = []
                tempLoss = []
                
                start_idx = np.random.randint(0, local_x.shape[0]-self.batch_size)
                
                batch_train_x = local_x[start_idx:(start_idx+self.batch_size)]
                batch_train_y = local_y[start_idx:(start_idx+self.batch_size)]
                
                '''
                if (local_x.shape[0] > self.batch_size):
                    
                    batch_train_x = local_x[start_idx:(start_idx+self.batch_size)]
                    batch_train_y = local_y[start_idx:(start_idx+self.batch_size)]
                    local_x = list(local_x)
                    local_y = list(local_y)
                    local_x = np.array(local_x[0:start_idx] + local_x[start_idx+self.batch_size:])
                    local_y = np.array(local_y[0:start_idx] + local_y[start_idx+self.batch_size:])
                    
                else:
                    done_epoch = True
                    batch_train_x = local_x
                    batch_train_y = local_y
                '''

                _, pred, stepLoss= self.sess.run([self.train_op, self.out, self.loss], feed_dict = {self.x: batch_train_x, self.y: batch_train_y})
                
                #print(stepLoss)     
                tempLoss.append(stepLoss)
                accuracy = 0
                for i,v in enumerate(batch_train_y):
                    if np.equal(np.argmax(v), np.argmax(pred[i])):
                        accuracy += 1
                accuracy /= len(batch_train_y)
                tempStore.append(accuracy*100)
                
                if step%50 == 0:
                    accuracyStore.append(np.mean(np.array(tempStore)))
                    accumLoss.append(np.mean(np.array(tempLoss)))
                    tempStore = []
                    tempLoss = []
                    
                    #epochLossArray.append(stepLoss)
                
                if (step%1000 == 0 and step != 0):
                
                    #Return accumulated epoch loss   
                    print("SAVING MODEL, {}th step".format(step))
                    self.save_model()
                    print(accuracyStore)
                    print("Averaged Accuracy {}".format(np.mean(np.array(accuracyStore))))
                    
                    #Display Accuracy Progression per Epoch
                    plt.plot(accuracyStore)
                    plt.ylabel("Accuracy Percentile")
                    plt.xlabel("Epoch step")
                    plt.show()
                    accuracyStore = []
                    
                    plt.plot(accumLoss[3:])
                    plt.ylabel("Loss")
                    plt.xlabel("Epoch step")
                    plt.show()
            
            return accumLoss
            
    def featurePredicton(self, input_set):
        #Returns prediction for the feature map on the second last layer
        
        return self.sess.run(self.img_extract, feed_dict = {self.x: input_set})

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
            if (np.equal(np.argmax(pred), np.argmax(y))):
                accuracy += 1
        accuracy /= data_x.shape[0]
        print("Final accuracy for test set: {}".format(accuracy*100))
        
    def save_model(self):
        
        with self.graph.as_default():
            self.saver.save(self.sess, "models/revisedModel.ckpt")

    