import numpy as np
import tensorflow as tf
from Preprocessing import save_expanded, encode_oneHot, load_embeddings, convert_format, closest_vectors
#import tensorflow.keras.datasets as datasets  
from model import zsModel
import matplotlib.pyplot as plt
import sklearn.manifold as fold

def main():


    train = True
    test = True
    
    #save_expanded()
    x_train = np.load("expandedTrain.npy")
    x_test = np.load("expandedTest.npy")
    
    zsl_dict = None
    train_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    zsl_classes = ['bicycle', 'helicopter', 'submarine']
    relevant_classes = train_classes + zsl_classes
    zsl_dict = load_embeddings("glove.6B.100d.txt", train_classes, zsl_classes)
    
    y_train = np.load("trainY.npy")
    y_test = np.load("testY.npy")
    
    hparams = tf.contrib.training.HParams(input_shape = (None, 50, 50, 3),
                                          output_shape = (None, 10),
                                          feature_shape = 100,
                                          batch_size = 100,
                                          filters = [[64,3, False, 'SAME', True],
                                                     [64,3, True, 'SAME', False],
                                                     [128,3, False, 'SAME', True],
                                                     [128,3, True, 'SAME', False],
                                                     [256,3, False, 'SAME', True],
                                                     [256,3, False, 'VALID', False],
                                                     ])
    
    newModel = zsModel(hparams, "ZSModel")
    newModel.load_weights("revisedModel.ckpt")
    print("GRAPH BUILT")
    
    #step = 0
    
    #closest_vectors("glove.6B.100d.txt", train_classes, zsl_dict)
    '''
    loss = newModel.train(x_train, y_train, steps = 5001)
    
    plt.plot(loss[2:])
    plt.ylabel("Loss (Accumulation per Epoch)")
    plt.xlabel("Step (per 50)")
    plt.show()
        
    newModel.save_model()
    '''
    
    newModel.accuracy_test(x_test, y_test)

    
    relevant_embeds = []
    
    for pred_class in relevant_classes:
        relevant_embeds.append(zsl_dict[pred_class]) #In order of relevant classes
    relevant_embeds = np.array(relevant_embeds)
    x_embed = fold.TSNE(n_components=2, init = 'pca', n_iter = 2000).fit_transform(relevant_embeds)

    reshaped_embed = np.zeros((2,len(relevant_classes)))
    
    for i,v in enumerate(x_embed):
        reshaped_embed[0][i] = v[0]
        reshaped_embed[1][i] = v[1]
    
    for i in range(reshaped_embed.shape[-1]):
        plt.scatter(reshaped_embed[0][i], reshaped_embed[1][i])
        plt.annotate(relevant_classes[i], (reshaped_embed[0][i], reshaped_embed[1][i]))
        
    plt.show()
    
    if test:
        totalImg = []
        totalLabel = []
        newModel.accuracy_test(x_test, y_test)
        
        #Read in images
        for zclass in zsl_classes:
            path = "zsImages/"
            img, label = convert_format(path+zclass+"s", zclass)
            totalImg += img
            totalLabel += label
            
        zsl_accuracies = dict()
        zsl_accuracies_k = dict()
        for zclass in zsl_classes:
            zsl_accuracies[zclass] = 0
            zsl_accuracies_k[zclass] = 0

        once = True
        for img, label in zip(totalImg, totalLabel):
            
            euc_distances = []
            features = newModel.featurePredicton(np.expand_dims(img, axis = 0))
            features = np.squeeze(features)
            
            #Visualize vectors with tsne dim reduction
            
            '''
            relevant_embeds = []
            for pred_class in relevant_classes:
                relevant_embeds.append(zsl_dict[pred_class])
            relevant_embeds.append(list(features)) #Maintain format

            relevant_embeds = np.array(relevant_embeds)
            print(relevant_embeds.shape)
            
            x_embed = fold.TSNE(n_components=2, init = 'pca', n_iter = 5000).fit_transform(relevant_embeds)
            reshaped_embed = np.zeros((2,len(relevant_classes)+1))
            
            #Reshape
            for i,v in enumerate(x_embed):
                reshaped_embed[0][i] = v[0]
                reshaped_embed[1][i] = v[1]
            reshaped_embed = np.array(reshaped_embed)
            
            #Plot prediction
            for i in range(reshaped_embed.shape[-1]):
                plt.scatter(reshaped_embed[0][i], reshaped_embed[1][i])
                if (i < len(relevant_classes)):
                    plt.annotate(relevant_classes[i], (reshaped_embed[0][i], reshaped_embed[1][i]))
            plt.show()
            '''
            
            #Minimize euclidean distance to nearest word vector
            for test_class in zsl_dict.keys():
                if once:
                    print(features, " NOOO")
                    print(zsl_dict[test_class])
                    once = False
                euc_distances.append(np.linalg.norm(features-zsl_dict[test_class]))
            euc_array =np.array(euc_distances)
            min_idx = np.argmin(euc_array)
            k_smallest = np.argpartition(euc_array, 3)[:3]
            print(k_smallest)
            
            #Calculuate accuracies
            for i,v in enumerate(zsl_dict.keys()):
                if (i == min_idx):
                    if (v == label):
                        zsl_accuracies[label] += 1
                        
            for i, v in enumerate(zsl_dict.keys()):
                if i in k_smallest:
                    if v == label:
                        zsl_accuracies_k[label] += 1
                
            if label == 'bicycle':
                plt.imshow(img)
                plt.show()
                for i,v in enumerate(zsl_dict.keys()):
                    if (i == min_idx):
                        print(v, " prediction.")
                    
        for key in zsl_accuracies.keys():
            zsl_accuracies[key] /= totalLabel.count(key)
            print("ZSL Accuracy for {} is: {}".format(key, zsl_accuracies[key]))
            zsl_accuracies_k[key] /= totalLabel.count(key)
            print("ZSL Accuracy (Top 3) for {} is {}".format(key, zsl_accuracies_k[key]))
    
if __name__ == "__main__":
    main()