
import os
import cv2
import numpy as np
from keras.applications.resnet import preprocess_input
from tempfile import TemporaryFile

directory = "C:\\Users\\zhelf\\OneDrive\\Desktop\\Grad School\\RBE595\\Week 12\\HW 7-8\\CroppedYale\\CroppedYale"
print("Working in: ", directory)
X_train = np.empty((1,192,168,1))
y_train = np.empty((1,1))
i = 0
for dir in os.listdir(directory):
    i += 1
    print(dir)
    for file in os.listdir(directory+"//"+dir):
        file = str(file)
        print(file)
        x = cv2.imread(directory+"//"+dir+"//"+file, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (168, 192))
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=-1)
        npfeatures = np.array(x)
        # print("features: ", npfeatures.shape, npfeatures)
        # print("train: ", X_train, X_train.shape)
        X_train = np.append(X_train, npfeatures, axis=0)
        # y_train = np.append(y_train, np.array([[i]]), axis=0)
        # print(y_train, y_train.shape)
X_train = X_train[1:]
print(X_train.shape)
# y_train = y_train[1:]
# print(y_train, y_train.shape)
# y_train_file = TemporaryFile()
# np.save("C:\\Users\\zhelf\\source\\repos\\PythonApplication1\\y_train_file", y_train, allow_pickle=True)
# X_train_file = TemporaryFile()
# np.save("C:\\Users\\zhelf\\source\\repos\\PythonApplication1\\X_train_file", X_train, allow_pickle=True)

