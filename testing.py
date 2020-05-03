import os, cv2
from mesonet_classifiers import *


pwd = os.path.dirname(__file__)
FOLDER_PATH_POSITIVE_SAMPLES = pwd + "./test"
FOLDER_PATH_NEGATIVE_SAMPLES = pwd + "./test"
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
        if classifier.predict(im) == 1:
            true_pos += 1
        else:
            false_neg += 1

for filename in os.listdir(FOLDER_PATH_NEGATIVE_SAMPLES):
    im = cv2.imread(os.path.join(FOLDER_PATH_NEGATIVE_SAMPLES, filename))
    if im is not None:
        if classifier.predict(im) == 0:
            true_neg += 1
        else:
            false_pos += 1

total = true_pos + true_neg + false_pos + false_neg
accuracy = (true_pos + true_neg) / total

print("Results:")
print("Accuracy: " + accuracy)
print("True positive: " + true_pos / total)
print("True negative: " + true_neg / total)
print("False positive: " + false_pos / total)
print("False negative: " + false_neg / total)