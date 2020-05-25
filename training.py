import os, cv2
from mesonet_classifiers import *
import numpy as np


pwd = os.path.dirname(__file__)
FOLDER_PATH_POSITIVE_SAMPLES = pwd + "./train_original"
FOLDER_PATH_NEGATIVE_SAMPLES = pwd + "./train_warped"
FOLDER_PATH_SAVE_WEIGHTS = pwd + "./8a_weights"


classifier = MesoInception4(learning_rate=0.001)
xs = []
ys = []

for filename in os.listdir(FOLDER_PATH_POSITIVE_SAMPLES):
    im = cv2.imread(os.path.join(FOLDER_PATH_POSITIVE_SAMPLES, filename))
    if im is not None:
        xs.append(im)
        ys.append(1)

for filename in os.listdir(FOLDER_PATH_NEGATIVE_SAMPLES):
    im = cv2.imread(os.path.join(FOLDER_PATH_NEGATIVE_SAMPLES, filename))
    if im is not None:
        xs.append(im)
        ys.append(0)

xs = np.array(xs)
ys = np.array(ys)

# classifier.fit(xs, ys) # class method 'fit' of MesoInception4 uses keras train_on_batch() function, but we want fit()
classifier.model.fit(xs, ys, batch_size=64, epochs=20)
classifier.save(FOLDER_PATH_SAVE_WEIGHTS)
