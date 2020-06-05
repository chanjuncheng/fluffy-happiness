import os, cv2, numpy as np
from mesonet_classifiers import MesoInception4


pwd = os.path.dirname(__file__)
FOLDER_PATH_POSITIVE_SAMPLES = pwd + "./test_imgs/positive"
FOLDER_PATH_NEGATIVE_SAMPLES = pwd + "./test_imgs/negative"
DIR_TO_WEIGHTS_FILE = pwd + "./8a_mesoinception4.h5"


classifier = MesoInception4()
classifier.load(DIR_TO_WEIGHTS_FILE)

# !!! in this context, positive = manipulated, negative = genuine
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0

# positive samples = genuine = negative in sensitivity context
for filename in os.listdir(FOLDER_PATH_POSITIVE_SAMPLES):
    im = cv2.imread(os.path.join(FOLDER_PATH_POSITIVE_SAMPLES, filename))
    if im is not None:
        im = np.expand_dims(im, axis=0)
        res = classifier.predict(im)[0][0]
        if res >= 0.5:
            false_pos += 1
        else:
            true_neg += 1

# negative samples = simulated deepfake = manipulated = positive in sensitivity context
for filename in os.listdir(FOLDER_PATH_NEGATIVE_SAMPLES):
    im = cv2.imread(os.path.join(FOLDER_PATH_NEGATIVE_SAMPLES, filename))
    if im is not None:
        im = np.expand_dims(im, axis=0)
        res = classifier.predict(im)[0][0]
        if res < 0.5:
            false_neg += 1
        else:
            true_pos += 1

total = true_pos + true_neg + false_pos + false_neg
accuracy = float(true_pos + true_neg) / total

print("Results:")
print("Accuracy: " + str(accuracy))
print("True positive: " + str(float(true_pos) / total))
print("True negative: " + str(float(true_neg) / total))
print("False positive: " + str(float(false_pos) / total))
print("False negative: " + str(float(false_neg) / total))
