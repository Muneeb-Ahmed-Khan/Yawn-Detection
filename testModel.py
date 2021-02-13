import cv2
from imutils import face_utils
import numpy as np
import argparse
import os
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", help="path to facial landmark predictor", default="shape_predictor_68_face_landmarks.dat")
ap.add_argument("-m", "--model", help="h5 Model", default="model/yawnModel.hdf5")
ap.add_argument("-v", "--video", help="Testing Video Path", default="videos/1.avi")
args = vars(ap.parse_args())



if os.path.isfile(args["shape_predictor"]):
	pass
else:
    print("No Facial Landmark Perdictor.")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

yawns = 0
yawn_status = False


model = load_model(args["model"])
cap = cv2.VideoCapture(args["video"])

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break
    # Our operations on the frame come here
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        pad = 3
        crop_image = image[miny - pad: maxy + pad, minx - pad: maxx + pad]

        roi = cv2.resize(crop_image, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0)

        # determine the probabilities of both "smiling" and "not similing"
        # then set the label accordingly
        (notSmiling, smiling) = model.predict(roi)[0]
        label = "Yawning" if smiling > notSmiling else "Normal"

        # show the Result
        cv2.putText(frame, label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()