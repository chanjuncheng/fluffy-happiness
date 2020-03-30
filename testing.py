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
                print(roi[0].shape)
                e = lib.get_face_mask(roi[0].shape[:2], point)
                cv2.imwrite('e.jpg', e)

                # print('face info ', face_info)
                cv2.imwrite('info.jpg', np.asarray(face_info[0][1]))

                a = lib.get_all_face_mask(roi[0].shape, face_info)
                b = lib.get_aligned_face_and_landmarks(im, face_info, 256, (0,0) )
                bimg = np.asarray(b[0][0])
                c = pi.aug(bimg, {'rotation_range': 10, 'zoom_range': 0.05, 'shift_range': 0.05, 'random_flip': 0.5}, [0.5, 1.5])
                d = lib.get_face_loc(a, front_face_detector, scale=0)

                cv2.imwrite('d.jpg', np.asarray(d))
                cv2.imwrite('c.jpg', c)
                cv2.imwrite('b.jpg', bimg)
                cv2.imwrite('a.jpg',a)
                cv2.imwrite('ori.jpg', im)
                cv2.imwrite('head.jpg', roi[0])
                rois.append(cv2.resize(roi[0], tuple(cfg.IMG_SIZE[:2])))
            # concatenate rois
            # vis_im(rois, 'tmp/vis.jpg')


        #     prob = solver.test(rois)

        #     # half largest prob value -> get mean
        #     prob = np.mean(np.sort(prob[:, 0])[np.round(sample_num / 2).astype(int):])
        #     if prob >= max_prob:
        #         max_prob = prob
        # prob = max_prob
    return rois

im = cv2.imread('obama.jpg')
im_test(im)
# face_info = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)  # list of faces in im
# print(face_info[0][0])
# cv2.imwrite('output.png',np.asarray(face_info))
# cv2.imwrite('out.jpg', img)
# cv2.waitKey()