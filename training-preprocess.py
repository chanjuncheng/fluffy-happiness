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

TRAIN_IMGS_FOLDER_PATH = pwd + "./train_tests"

front_face_detector = dlib.get_frontal_face_detector()
lmark_predictor = dlib.shape_predictor(pwd + './dlib_model/shape_predictor_68_face_landmarks.dat')

def preprocess(im):

    image_size = im.shape[1], im.shape[0]
    faces = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)  # list of tuples of (transformation matrix, landmark point) of identified faces
    
    if len(faces) == 0:
        # logging.warning('No faces are detected.')
        return None

    # logging.info('{} faces are detected.'.format(len(faces)))

    trans_matrix, point = faces[0] # take only the first face found

    size = random.randint(RESIZE_MIN, RESIZE_MAX + 1) # align face to a size k x k where k is a number between RESIZE_MIN and RESIZE_MAX

    # Figure 2 in CVPRW2019_Face_Artifacts

    face = cv2.warpAffine(im, trans_matrix * size, (size, size))
    face = cv2.GaussianBlur(face, (5,5), 0)

    # xj resizing code
    # face_imgs, face_landmark_coords = lib.get_aligned_face_and_landmarks(im, faces, 256, (0,0))
    # face_img = face_imgs[0]
    # face = cv2.resize(face_img, (size, size))

    warped_im = np.copy(im)
    warped_im = cv2.warpAffine(face, trans_matrix*size, image_size, warped_im, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT)

    # Figure 3 in CVPRW2019_Face_Artifacts

    # get mask of facial area and surrounds, convert to grayscale
    mask = lib.get_face_mask(im.shape[:2], point)
    mask_inv = cv2.bitwise_not(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_BGR2GRAY)

    area_surrounding = cv2.bitwise_or(im, im, mask = mask_inv)
    area_surrounding = area_surrounding[0:im.shape[0], 0:im.shape[1]]

    area_facial = cv2.bitwise_or(warped_im, warped_im, mask = mask)
    area_facial = area_facial[0:im.shape[0], 0:im.shape[1]]

    output_im = area_surrounding + area_facial

    return output_im


def batch_preprocess(ims):

    output_ims = []
    failed_indices = []

    for i in range(len(ims)):
        output = preprocess(ims[i])
        if output is not None:
            output = cv2.resize(output, (256,256))
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
        new_filename = filenames[i][:-4] + "_warped" + filenames[i][-4:]
        path = os.path.join(folder_path, new_filename)
        cv2.imwrite(path, ims[i])



def ori_batch_imwrite(folder_path, filenames, ims):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in range(len(ims)):
        ori_filename = filenames[i][:]
        path = path = os.path.join(folder_path, ori_filename)
        cv2.imwrite(path, ims[i])


print("Reading all images from directory...")

ims, filenames, failed = batch_imread(TRAIN_IMGS_FOLDER_PATH)

ori = []
for img in ims:
    img = cv2.resize(img, (256,256))
    ori.append(img)
    
ori_batch_imwrite(os.path.join(TRAIN_IMGS_FOLDER_PATH, "../train_original"), filenames, ori)


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

batch_imwrite(os.path.join(TRAIN_IMGS_FOLDER_PATH, "../train_warped"), filenames, output_ims)

print("OK")
print("Done.")


# TODO: increase diversity by introducing different masks
# TODO: crop ROI of [y0 - y^0, x0 - x^0, y1 + y^1, x1 + x^1] where the minumum bounding box b is expanded in each direction by random between
# [0, h/5] and [0, w/8] where h, w are height and width of b. then resize to 224 x 224 for final training data
