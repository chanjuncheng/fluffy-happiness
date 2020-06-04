import os, cv2, numpy as np
from mesonet_classifiers import MesoInception4


pwd = os.path.dirname(__file__)
FOLDER_PATH_POSITIVE_SAMPLES = pwd + "./train_imgs/positive"
FOLDER_PATH_NEGATIVE_SAMPLES = pwd + "./train_imgs/negative"
DIR_TO_WEIGHTS_FILE = pwd + "./8a_mesoinception4.h5"


classifier = MesoInception4(learning_rate=0.001)
xs = []
ys = []

for filename in os.listdir(FOLDER_PATH_POSITIVE_SAMPLES):
    im = cv2.imread(os.path.join(FOLDER_PATH_POSITIVE_SAMPLES, filename))
    if im is not None:
        xs.append(im)
        ys.append(0) # positive samples = genuine

for filename in os.listdir(FOLDER_PATH_NEGATIVE_SAMPLES):
    im = cv2.imread(os.path.join(FOLDER_PATH_NEGATIVE_SAMPLES, filename))
    if im is not None:
        xs.append(im)
        ys.append(1) # negative samples = simulated deepfake = manipulated

xs = np.array(xs)
ys = np.array(ys)

classifier.model.fit(xs, ys, batch_size=64, epochs=20)
classifier.save(DIR_TO_WEIGHTS_FILE)
