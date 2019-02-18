#import tensorflow.keras.datasets as datasets  
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def save_expanded():

    #Load directly from cifar10
    training_set, testing_set = datasets.cifar10.load_data()
    x_train, _ = training_set
    x_test, _ = testing_set
    
    x_train_new = []
    x_test_new = []
    
    #Resize from 32x32 to 150x150, interpolation for enhanced resolution
    
    for img in (x_train):
        newImage = cv2.resize(img, dsize = (50, 50), interpolation = cv2.INTER_CUBIC)
        newImage = (newImage-np.min(newImage))/(np.max(newImage) - np.min(newImage)) #Normalize
        x_train_new.append(newImage)
        
    x_train_new = np.array(x_train_new)
    
    for img in (x_test):
        newImage = cv2.resize(img, dsize = (50, 50), interpolation = cv2.INTER_CUBIC)
        newImage = (newImage-np.min(newImage))/(np.max(newImage) - np.min(newImage))
        x_test_new.append(newImage)
        
    x_test_new = np.array(x_test_new)
    
    #Demo plot of resized image
    plt.imshow(x_train_new[0])
    
    #Save new image arrays
    np.save("expandedTrain.npy", x_train_new)
    np.save("expandedTest.npy", x_test_new)

def convert_format(path, label):
    
    #Reads all the images in a given file directory, and preprocesses them into use for the network
    files = os.listdir(path)
    imgList = []
    labelList = []
    print("{} and {}".format(path, files[0]))
    for i,file in enumerate(files):
        img = cv2.imread(os.path.join(path, file), 1)
        img = cv2.resize(img, dsize = (50, 50), interpolation = cv2.INTER_CUBIC)
        imgList.append(img)
        labelList.append(label)
    return imgList, labelList

def encode_oneHot(labels, idx):
    
    encoded_labels = []
    for label in labels:
        zero_vec = np.zeros((idx))
        zero_vec[label] = 1
        encoded_labels.append(zero_vec)
    return np.array(encoded_labels)

def load_embeddings(path, wanted_embs, zero_shot_classes, save_embed = False):
    
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
    
    for word_part in word_sep:
        temp_array = word_part.split()
        if temp_array[0] in total_classes:
            embeddingList = list(map(float, temp_array[1:]))
            word_dict[temp_array[0]] = embeddingList
    
    if save_embed:
        
        embed_matrix = np.array([], dtype = np.float32).reshape(0, 100)
        for word in wanted_embs:
            embed_matrix = np.vstack([embed_matrix, np.array(word_dict[word])])
            
        np.save("embedding_matrix.npy", embed_matrix)
        
    return word_dict
    
def closest_vectors(path, train_classes, embed_dict):
    
    embed_file = open(path, 'r')
    text = embed_file.read()
    word_sep = text.split('\n')
    
    #Last element is blank
    word_sep.pop()
    distance_array = []
    
    for tclass in train_classes:
        for word_part in word_sep:
            temp_array = word_part.split()

            distance_array.append(np.linalg.norm(np.array(embed_dict[tclass]) - np.array(temp_array[1:]).astype(np.float64)))
            closest_vectors = []
        for _ in range(20):
            smallest_idx = np.argmin(np.array(distance_array))
            closest_vectors.append(smallest_idx)
            distance_array[smallest_idx] = float('inf')
        closest_vectors = ' '.join([word_sep[i].split()[0] for i in closest_vectors])
        print("{}'s closest 20 vectors are:{}".format(tclass, closest_vectors))
        closest_vectors = []
        distance_array = []
        
            
    