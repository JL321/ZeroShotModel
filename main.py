import numpy as np
import tensorflow as tf
from Preprocessing import save_expanded, encode_oneHot, load_embeddings, convert_format
import tensorflow.keras.datasets as datasets  
from model import zsModel
import matplotlib.pyplot as plt

def main():

    #save_expanded()
    x_train = np.load("expandedTrain.npy")
    x_test = np.load("expandedTest.npy")
    
    train_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    zsl_classes = ['zebra', 'canoe', 'helicopter', 'fox']
    zsl_dict = load_embeddings("glove.6B.100d.txt", train_classes, zsl_classes)
    
    (_, y_train), (_, y_test) = datasets.cifar10.load_data()
    y_train = encode_oneHot(y_train, 10)
    y_test = encode_oneHot(y_test, 10)
    
    hparams = tf.contrib.training.HParams(input_shape = (None, 50, 50, 3),
                                          output_shape = (None, 10),
                                          feature_shape = 100,
                                          batch_size = 16,
                                          filters = [[64,3, True, 'SAME'],
                                                     [128,3, False, 'SAME'],
                                                     [128,3, False, 'SAME'],
                                                     [128,3, False, 'VALID'],
                                                     [256,3, True, 'VALID'],
                                                     [256,3, False, 'VALID']
                                                     ])
    
    newModel = zsModel(hparams, "ZSModel")
    #newModel.load_weights("newModel.ckpt")
    print("GRAPH BUILT")

    loss = newModel.train(x_train, y_train, 2)
    plt.plot(loss)
    plt.ylabel("Loss (Accumulation per Epoch)")
    plt.xlabel("Step")
    plt.show()
    
    newModel.accuracy_test(x_train, y_train)
    
    imgList = convert_format("zsImages")
    for img in imgList:
        euc_distances = []
        features = newModel.featurePredicton(np.expand_dims(img, axis = 0))
        for test_class in zsl_dict.keys():
            euc_distances.append(np.linalg.norm(features-zsl_dict[test_class]))
        min_idx = np.argmin(np.array(euc_distances))
        for i,v in enumerate(zsl_dict.keys()):
            if (i == min_idx):
                print("Prediction: {}".format(v))
        plt.imshow(img)
        plt.show()
    
    
if __name__ == "__main__":
    main()
