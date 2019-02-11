import tensorflow.keras.datasets as datasets  
import numpy as np
import matplotlib.pyplot as plt
import cv2

def save_expanded():

    #Load directly from cifar10
    training_set, testing_set = datasets.cifar10.load_data()
    x_train, y_train = training_set
    x_test, y_test = testing_set
    
    x_train_new = []
    x_test_new = []
    
    #Resize from 32x32 to 150x150, interpolation for enhanced resolution
    
    for img in (x_train):
        newImage = cv2.resize(img, dsize = (50, 50), interpolation = cv2.INTER_CUBIC)
        x_train_new.append(newImage)
        
    x_train_new = np.array(x_train_new)
    
    for img in (x_test):
        newImage = cv2.resize(img, dsize = (50, 50), interpolation = cv2.INTER_CUBIC)
        x_test_new.append(newImage)
        
    x_test_new = np.array(x_test_new)
    
    #Demo plot of resized image
    plt.imshow(x_train_new[0])
    
    #Save new image arrays
    np.save("expandedTrain.npy", x_train_new)
    np.save("expandedTest.npy", x_test_new)

def encode_oneHot(labels, idx):
    
    encoded_labels = []
    for label in labels:
        zero_vec = np.zeros((idx))
        zero_vec[label] = 1
        encoded_labels.append(zero_vec)
    return encoded_labels

def load_embeddings(path, wanted_embs, zero_shot_classes):
    
    """
    path: Path of word embeddings
    wanted_embs: String list of training classes of zero shot model
    zero_shot_classes: String list of all classes in zero shot model
    """
    
    total_classes = wanted_embs+zero_shot_classes
    embed_file = open(path, 'r')
    text = embed_file.read()
    word_sep = text.split('\n')
    
    #Last element is blank
    word_sep.pop()
    word_dict = {}
    embed_matrix = np.array([], dtype = np.float32).reshape(0, 100)
    
    for word_part in word_sep:
        temp_array = word_part.split()
        if temp_array[0] in total_classes:
            embeddingList = list(map(float, temp_array[1:]))
            word_dict[temp_array[0]] = embeddingList
    
    for word in wanted_embs:
        embed_matrix = np.vstack([embed_matrix, np.array(word_dict[word])])
        
    print(embed_matrix.shape," C")
    np.save("embedding_matrix.npy", embed_matrix)
    return word_dict
    
        
        
    
        
    
    