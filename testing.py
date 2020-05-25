import os, cv2
from mesonet_classifiers import *
import numpy as np


pwd = os.path.dirname(__file__)
FOLDER_PATH_POSITIVE_SAMPLES = pwd + "./test_original"
FOLDER_PATH_NEGATIVE_SAMPLES = pwd + "./test_warped"
FOLDER_PATH_WEIGHTS = pwd + "./8a_weights"


classifier = MesoInception4()
classifier.load(FOLDER_PATH_WEIGHTS)

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0

for filename in os.listdir(FOLDER_PATH_POSITIVE_SAMPLES):
    im = cv2.imread(os.path.join(FOLDER_PATH_POSITIVE_SAMPLES, filename))
    if im is not None:
        im = np.expand_dims(im, axis=0)
        if classifier.predict(im) == 1:
            true_pos += 1
        else:
            false_neg += 1

for filename in os.listdir(FOLDER_PATH_NEGATIVE_SAMPLES):
    im = cv2.imread(os.path.join(FOLDER_PATH_NEGATIVE_SAMPLES, filename))
    if im is not None:
        im = np.expand_dims(im, axis=0)
        if classifier.predict(im) == 0:
            true_neg += 1
        else:
            false_pos += 1

total = true_pos + true_neg + false_pos + false_neg
accuracy = (true_pos + true_neg) / total

print("Results:")
print("Accuracy: " + str(accuracy))
print("True positive: " + str(true_pos / total))
print("True negative: " + str(true_neg / total))
print("False positive: " + str(false_pos / total))
print("False negative: " + str(false_neg / total))