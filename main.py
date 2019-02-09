import numpy as np
import tensorflow as tf
from Preprocessing import save_expanded, encode_oneHot
import tensorflow.keras.datasets as datasets  
from model import zsModel
import matplotlib.pyplot as plt

def main():

    #save_expanded()
    x_train = np.load("expandedTrain.npy")
    x_test = np.load("expandedTest.npy")
    
    (_, y_train), (_, y_test) = datasets.cifar10.load_data()
    y_train = encode_oneHot(y_train, 10)
    y_test = encode_oneHot(y_test, 10)
    
    hparams = tf.contrib.training.HParams(input_shape = (None, 50, 50, 3),
                                          output_shape = (None, 10),
                                          feature_shape = 100,
                                          batch_size = 16,
                                          filters = [[64,3, True],
                                                     [128,3, False],
                                                     [128,3, False],
                                                     [128,3, True],
                                                     [256,3, False],
                                                     ])
    
    newModel = zsModel(hparams, "ZSModel")
    print("GRAPH BUILT")
    loss = newModel.trainModel(x_train, y_train, 20)
    plt.plot(loss)
    plt.ylabel("Loss (Accumulation per Epoch)")
    plt.xlabel("Epoch")
    
if __name__ == "__main__":
    main()