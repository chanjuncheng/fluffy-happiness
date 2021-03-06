import sys, os, cv2, dlib, numpy as np
from py_utils.face_utils import lib
from mesonet_classifiers import MesoInception4

'''
Given a video (.mp4) or image file (.jpg, .png), predict if the content was modified with DeepFake using the trained model.
'''

def preprocess(im): # refactored out of preprocess.py to accommodate the requirements of PyQt5

    '''
    Given an input image, preprocess it the same way as training/testing samples and return the output image.
    '''

    # image size = input size to model
    IMG_SIZE = 256

    front_face_detector = dlib.get_frontal_face_detector()
    lmark_predictor = dlib.shape_predictor('./dlib_model/shape_predictor_68_face_landmarks.dat')
    
    # list of tuples of (transformation matrix, landmark point) of identified faces
    faces = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)

    if len(faces) == 0:
        return None

    # PART REMOVED DUE TO LOWER ACCURACY
    # take only the first face found
    # trans_matrix, _ = faces[0]
    # face = cv2.warpAffine(im, trans_matrix * FACE_SIZE, (FACE_SIZE, FACE_SIZE))

    # simply resizing without cropping in to facial region, as accuracy has been found to have increased
    im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
    return im


def predict(filepath):

    '''
    Given a file path to a piece of content in the required format (.mp4, .jpg, .png), analyzes it and returns a prediction
    of whether the content has been modified using the Deepfake algorithm.
    If the result is > 0.5, it was predicted to have been modified, vice versa.
    '''

    FOLDER_PATH_WEIGHTS = "./8a_mesoinception4.h5"
    classifier = MesoInception4()
    classifier.load(FOLDER_PATH_WEIGHTS)

    # check if valid file
    is_file = os.path.isfile(filepath)
    is_video = filepath[-4:] == ".mp4"
    is_image = filepath[-4:] == ".jpg" or filepath[-4:] == ".png"

    if not is_file or (not is_video and not is_image):
        raise Exception("Invalid file format.")

    # start prediction

    res = -1

    if is_video:

        # positive = manipulated, negative = genuine
        pos_count = 0
        neg_count = 0

        stream = cv2.VideoCapture(filepath)

        if not stream.isOpened():
            raise Exception("Video file cannot be opened.")

        is_running, im = stream.read()

        while is_running:

            face = preprocess(im)
            if face is not None: # if no face found, skip current frame
                face = np.expand_dims(face, axis=0)
                res = classifier.predict(face)[0][0]
                if res >= 0.5:
                    pos_count += 1
                else:
                    neg_count += 1
            is_running, im = stream.read()

        if pos_count > neg_count:
            res = 1
        else: # including situations where neg_count > pos_count, neg_count == pos_count, or when both are 0 (none of the frames contain faces)
            res = 0

    if is_image:

        im = cv2.imread(filepath)
        face = preprocess(im)
        if face is not None:
            face = np.expand_dims(face, axis=0)
            res = classifier.predict(face)[0][0]

    # logging prediction and returning result

    # print("Prediction: " + "This video/image has been manipulated" if res > 0.5 else "This video/image is unmodified")

    if res > 0.5:
        return True
    else:
        return False

    # Assumption made: if no face is found, treat content as unmodified as DeepFake only works on facial regions


if __name__ == "__main__":
    filepath = sys.argv[1]
    res = predict(filepath)
    print(res)