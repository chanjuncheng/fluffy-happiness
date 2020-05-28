import os, cv2, dlib
from py_utils.face_utils import lib
from mesonet_classifiers import *
import numpy as np



'''
Given a video (.mp4) or image file (.jpg), predict if the content was modified with DeepFake using the trained model
'''

# pwd = os.path.dirname(__file__)
# FOLDER_PATH_WEIGHTS = pwd + "./8a_weights"
# FACE_SIZE = 128
#
# front_face_detector = dlib.get_frontal_face_detector()
# lmark_predictor = dlib.shape_predictor(pwd + './dlib_model/shape_predictor_68_face_landmarks.dat')


def extract_face(im):
    pwd = os.path.dirname(__file__)
    FACE_SIZE = 128

    front_face_detector = dlib.get_frontal_face_detector()
    # lmark_predictor = dlib.shape_predictor(pwd + './dlib_model/shape_predictor_68_face_landmarks.dat')
    lmark_predictor = dlib.shape_predictor('./dlib_model/shape_predictor_68_face_landmarks.dat')
    
    faces = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)  # list of tuples of (transformation matrix, landmark point) of identified faces

    if len(faces) == 0:
        return None

    # take only the first face found
    trans_matrix, _ = faces[0]
    face = cv2.warpAffine(im, trans_matrix * FACE_SIZE, (FACE_SIZE, FACE_SIZE))

    return face

def prediction(filepath):
    # pwd = os.path.dirname(__file__)
    # FOLDER_PATH_WEIGHTS = pwd + "./8a_weights"
    FOLDER_PATH_WEIGHTS = "./8a_weights"
    classifier = MesoInception4()
    classifier.load(FOLDER_PATH_WEIGHTS)

    ######################
    # GET INPUT DIR HERE #
    ######################
    # filepath = "./niccage.jpg"

    is_file = os.path.isfile(filepath)
    is_video = filepath[-4:] == ".mp4"
    is_image = filepath[-4:] == ".jpg" or filepath[-4:] == ".png"


    if not is_file or (not is_video and not is_image):
        raise Exception("Invalid file format.")

    res = -1

    if is_video:

        pos_count = 0
        neg_count = 0

        stream = cv2.VideoCapture(filepath)

        if not stream.isOpened():
            raise Exception("Video file cannot be opened.")

        is_running, im = stream.read()

        while is_running:
            face = extract_face(im)
            if face is not None: # if no face found, skip current frame
                face = cv2.resize(face, (256,256))
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
        face = extract_face(im)
        if face is not None:
            face = cv2.resize(face, (256,256))
            face = np.expand_dims(face, axis=0)
            res = classifier.predict(face)[0][0]
            if res >= 0.5:
                res = 1
            else:
                res = 0

    print("Prediction: " + "This video/image has been manipulated" if res == 0 else "This video/image is unmodified")
    if res == 0:
        return True
    else:
        return False

    # Assumption made: if no face is found, treat content as unmodified as DeepFake only works on facial regions
