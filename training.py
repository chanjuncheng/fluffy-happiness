import os, cv2
from mesonet_classifiers import *


pwd = os.path.dirname(__file__)
FOLDER_PATH_POSITIVE_SAMPLES = pwd + "./training/warped"
FOLDER_PATH_NEGATIVE_SAMPLES = pwd + "./training/original"
FOLDER_PATH_SAVE_WEIGHTS = pwd + "./8a_cfgs"


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

# classifier.fit(xs, ys) # class method 'fit' of MesoInception4 uses keras train_on_batch() function, but we want fit()
classifier.model.fit(xs, ys, batch_size=64, epochs=20)
classifier.save(os.path.join(FOLDER_PATH_SAVE_WEIGHTS, "./mesoinception4.h5"))
