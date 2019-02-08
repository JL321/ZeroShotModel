import numpy as np
import tensorflow as tf
from Preprocessing import save_expanded
import tensorflow.keras.datasets as datasets  
from model import zsModel

def main():

    x_train = np.load("expandedTrain.npy")
    x_test = np.load("expandedTest.npy")
    
    (_, y_train), (_, y_test) = datasets.cifar10.load_data()
    
    hparams = tf.contrib.training.HParams(input_shape = (None, 150, 150, 3),
                                          output_shape = 10,
                                          feature_shape = 100,
                                          batch_size = 32,
                                          filters = [[64,3, False],
                                                     [64,3, True],
                                                     [128,3, False],
                                                     [256,3, True],
                                                     [512,3, False],
                                                     [512,3, True],
                                                     ])
    
    newModel = zsModel(hparams, "ZSModel")
    newModel.build_graph()
    print("GRAPH BUILT")
    newModel.train(x_train, y_train, 50)
    
if __name__ == "__main__":
    main()