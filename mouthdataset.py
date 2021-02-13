import glob
import cv2

from imutils import face_utils
import numpy as np
import argparse
import os
import imutils
import dlib
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", help="path to facial landmark predictor", default="shape_predictor_68_face_landmarks.dat")
# ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

if os.path.isfile(args["shape_predictor"]):
	pass
else:
    print("No Facial Landmark Perdictor.")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


images = [file for file in glob.glob('dataset/no-yawn/*.jpg')]
images_noYawn = [image.replace("\\", "/") for image in images]
for (i, image) in enumerate(images_noYawn):
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image, 0)
    orig = image
    image = imutils.resize(image, width=500)

    # detect faces in the grayscale image
    rects = detector(image, 1)
    for f in rects:
        shape_1 = predictor(image, f)
        shape = face_utils.shape_to_np(shape_1)

        xmouthpoints = [shape_1.part(x).x for x in range(48,67)]
        ymouthpoints = [shape_1.part(x).y for x in range(48,67)]
        maxx = max(xmouthpoints)
        minx = min(xmouthpoints)
        maxy = max(ymouthpoints)
        miny = min(ymouthpoints) 

        # to show the mouth properly pad both sides
        pad = 1
        crop_image = image[miny - pad: maxy + pad, minx - pad: maxx + pad]
        
        # cv2.imshow('mouth',crop_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite("dataset_new/no-yawn/" + str(i) + ".jpg", crop_image)

















images = [file for file in glob.glob('dataset/yawn/*.jpg')]
images_noYawn = [image.replace("\\", "/") for image in images]
for (i, image) in enumerate(images_noYawn):
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image, 0)
    orig = image
    image = imutils.resize(image, width=500)

    # detect faces in the grayscale image
    rects = detector(image, 1)
    for f in rects:
        shape_1 = predictor(image, f)
        shape = face_utils.shape_to_np(shape_1)

        xmouthpoints = [shape_1.part(x).x for x in range(48,67)]
        ymouthpoints = [shape_1.part(x).y for x in range(48,67)]
        maxx = max(xmouthpoints)
        minx = min(xmouthpoints)
        maxy = max(ymouthpoints)
        miny = min(ymouthpoints) 

        # to show the mouth properly pad both sides
        pad = 1
        crop_image = image[miny - pad: maxy + pad, minx - pad: maxx + pad]
        # cv2.imshow('mouth',crop_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite("dataset_new/yawn/" + str(i) + ".jpg", crop_image)
