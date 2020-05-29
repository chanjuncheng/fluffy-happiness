import tensorflow as tf
from resolution_network import ResoNet
from solver import Solver
from easydict import EasyDict as edict
import cv2, yaml, os, dlib
from py_utils.vis import vis_im
import numpy as np
from py_utils.face_utils import lib
from py_utils.vid_utils import proc_vid as pv
from py_utils.img_utils import proc_img as pi
import logging
import random

'''
Training designed to only work on images
'''

pwd = os.path.dirname(__file__)

# minimum and maximum range for facial region to be resized to (pixels x pixels), before being blurred and affine warped back to the original image
RESIZE_MIN = 64
RESIZE_MAX = 128

IMG_SIZE = 224

TRAIN_IMGS_FOLDER_PATH = pwd + "./train_tests"

front_face_detector = dlib.get_frontal_face_detector()
lmark_predictor = dlib.shape_predictor(pwd + './dlib_model/shape_predictor_68_face_landmarks.dat')

def preprocess(im):

    faces = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)  # list of tuples of (transformation matrix, landmark point) of identified faces
    
    if len(faces) == 0:
        # logging.warning('No faces are detected.')
        return None

    # logging.info('{} faces are detected.'.format(len(faces)))

    _, point = faces[0] # take only the first face found

    # crop image, after randomly expanding ROI (minimum bounding box b) in each direction between
    # [0, h/5] and [0, w/8] where h, w are height and width of b. then resize to 224 x 224 for final training data
    rois, _ = lib.cut_head([im], point, random.randint(0, 10))
    cropped_output_im = cv2.resize(rois[0], (IMG_SIZE, IMG_SIZE))

    return cropped_output_im


def batch_preprocess(ims):

    output_ims = []
    failed_indices = []

    for i in range(len(ims)):
        output = preprocess(ims[i])
        if output is not None:
            output_ims.append(output)
        else: # no face found
            failed_indices.append(i)

    return output_ims, failed_indices


def batch_imread(folder_path):

    ims = []
    failed_indices = []
    filenames = os.listdir(folder_path)

    for i in range(len(filenames)):
        im = cv2.imread(os.path.join(folder_path, filenames[i]))
        if im is not None:
            ims.append(im)
        else:
            failed_indices.append(i)

    return ims, filenames, failed_indices


def batch_imwrite(folder_path, filenames, ims):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in range(len(ims)):
        new_filename = filenames[i][:-4] + "_original" + filenames[i][-4:]
        path = os.path.join(folder_path, new_filename)
        cv2.imwrite(path, ims[i])


if __name__ == "__main__":
    
    print("Reading all images from directory...")

    ims, filenames, failed = batch_imread(TRAIN_IMGS_FOLDER_PATH)

    if len(failed) > 0:
        print("Some images failed to be read:")
        for i in failed:
            print(filenames[i])
    else:
        print("OK")

    # remove all filenames of failed files
    for i in range(len(filenames)-1, -1, -1):
        if i in failed:
            del filenames[i]

    print("Start preprocessing...")

    output_ims, failed = batch_preprocess(ims)

    if len(failed) > 0:
        print("Some files were skipped (no face found):")
        for i in failed:
            print(filenames[i])
    else:
        print("OK")

    # remove all filenames of failed files
    for i in range(len(filenames)-1, -1, -1):
        if i in failed:
            del filenames[i]

    print("Writing to output images...")

    batch_imwrite(os.path.join(TRAIN_IMGS_FOLDER_PATH, "./original"), filenames, output_ims)

    print("OK")
    print("Done.")
