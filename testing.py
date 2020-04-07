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
from py_8a_utils import affine_warp

pwd = os.path.dirname(__file__)
front_face_detector = dlib.get_frontal_face_detector()
lmark_predictor = dlib.shape_predictor(pwd + './dlib_model/shape_predictor_68_face_landmarks.dat')
cfg_file = 'cfgs/res50.yml'
with open(cfg_file, 'r') as f:
    cfg = edict(yaml.load(f))
sample_num = 10



def im_test(im):
    face_info = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)  # list of faces in im
    # Samples
   
    if len(face_info) == 0:
        logging.warning('No faces are detected.')
        prob = -1  # we ignore this case
        rois = []
    else:
        # Check how many faces in an image
        logging.info('{} faces are detected.'.format(len(face_info)))
        max_prob = -1
        rois = []
        # If one face is fake, the image is fake
        for _, point in face_info:
            # rois = []

            # get list of resized region of interest (faces)
            # repeated for sample_num=10 times with diff seed value
            for i in range(sample_num):
                roi, _ = lib.cut_head([im], point, i)   # roi = cropped im; _ = location of coordinates
                # print(roi)
                print('roi[0].shape: ', roi[0].shape)
                # e = lib.get_face_mask(roi[0].shape[:2], point)
                # cv2.imwrite('e.jpg', e)

                # print('face info ', face_info)
                cv2.imwrite('info.jpg', np.asarray(face_info[0][1]))

                a = lib.get_all_face_mask(roi[0].shape, face_info)
                b = lib.get_aligned_face_and_landmarks(im, face_info, 256, (0,0) )
                bimg = np.asarray(b[0][0])
                c = pi.aug(bimg, {'rotation_range': 10, 'zoom_range': 0.05, 'shift_range': 0.05, 'random_flip': 0.5}, [0.5, 1.5])
                d = lib.get_face_loc(a, front_face_detector, scale=0)

                # cv2.imwrite('d.jpg', np.asarray(d))
                # cv2.imwrite('c.jpg', c)
                # cv2.imwrite('b.jpg', bimg)
                # cv2.imwrite('a.jpg',a)
                # cv2.imwrite('ori.jpg', im)
                # cv2.imwrite('head.jpg', roi[0])
                rois.append(cv2.resize(roi[0], tuple(cfg.IMG_SIZE[:2])))

    return rois



im = cv2.imread('obama.jpg')

# face info = [[trans_matrix, alignment points], [trans_matrix, alignment points], ...]
face_info = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)
# b = [[image of detected faces], [landmark coordinates of detected faces]]
b = lib.get_aligned_face_and_landmarks(im, face_info, 256, (0,0) )

# self reference only: draw landmarks point on face
# points1 = np.asarray(b[1])
# for landmark in points1:
#     for x,y in landmark:
#         cv2.circle(b[0][0], (int(x), int(y)), 1, (0,0,0), 10)      
# cv2.imwrite('landmarks.jpg', b[0][0])



trans_matrix = face_info[0][0]
size = 64

face = cv2.resize(b[0][0], (size, size))
# cv2.imwrite('face.jpg', face)
face = cv2.GaussianBlur(face, (5,5), 0)
# cv2.imwrite('face2.jpg', face)

new_image = np.copy(im)
cv2.imwrite('face3.jpg', new_image)

image_size = im.shape[1], im.shape[0]
# cv2.warpAffine(face, trans_matrix*size, image_size, new_image, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT)
new_image = affine_warp.affineWarp(face, trans_matrix, image_size, new_image, True)
new_image = cv2.resize(new_image, image_size) # resize written here for now, plan to move into affineWarp
cv2.imwrite('face4.jpg', new_image)
