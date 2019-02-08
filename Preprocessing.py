import tensorflow.keras.datasets as datasets  
import numpy as np
import matplotlib.pyplot as plt
import cv2

training_set, testing_set = datasets.cifar10.load_data()
x_train, y_train = training_set
x_test, y_test = testing_set

path = "Emojis/glove.6B.100d.txt"
new_img = cv2.resize(x_train[0], dsize = (225, 225))
plt.imshow(new_img)


