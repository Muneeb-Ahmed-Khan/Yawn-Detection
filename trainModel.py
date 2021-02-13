# import packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from imutils import paths
from Model import CNN
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

# initialize the list of data and labels
data = []
labels = []


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="path to dataset", default="dataset")
ap.add_argument("-m", "--model", help="hdf5 Model", default="model/yawnModel.hdf5")
ap.add_argument("-p", "--plot", help="path to save plot", default="plot.jpg")
args = vars(ap.parse_args())


# loop over the input images
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    label = imagePath.split(os.path.sep)[-2]
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dim = (28, 28)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    #image = imutils.resize(image, width = 28, height=28)
    print(image.shape)
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the label list
    label = imagePath.split(os.path.sep)[-2]
    label = "yawning" if label == "yawn" else "not_yawning"
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# account for skew in the labeled data
classTotals = labels.sum(axis = 0)
classWeight = classTotals.max() / classTotals

# partition the data into training and testing splits using 80% of the data
# for training and remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.20, train_size= 0.80, stratify = labels, random_state = 42)

# initialize the model
print("[INFO] compiling model...")
model = CNN.build(width = 28, height = 28, depth = 1, classes = 2)
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size = 64, epochs = 15, verbose = 1)
model.save(args["model"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 64)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), target_names = le.classes_))

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, 15), H.history["accuracy"], label = "accuracy")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label = "val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args['plot'])
plt.show()
