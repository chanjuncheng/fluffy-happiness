import os
import unittest
import cv2
from preprocess import *
import predict as prediction

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.folder_path = './unit_test_samples/test'
        self.face_folder_path = './unit_test_samples/test_face'
        self.without_face_folder_path = './unit_test_samples/test_without_face'
        self.face_folder_video_path = './unit_test_samples/test_face_video'
        self.without_face_folder_video_path = './unit_test_samples/test_without_face_video'
        self.errors = []

    def tearDown(self):
        self.assertEqual([], self.errors)
    
    # test batch reading image
    def test_batch_imread(self):
        ims, filenames, failed = batch_imread(self.folder_path)
        total_length = len(ims) + len(failed)
        self.assertEqual(total_length, len(os.listdir(self.folder_path)))

    # test preprocess of images
    def test_preprocess(self):
        ims, filenames, failed = batch_imread(self.folder_path)
        for i in range(len(ims)):
            im = cv2.imread(os.path.join(self.folder_path, filenames[i]))
            res = preprocess(im)
            if res is not None:
                self.assertEqual(res.shape, (256, 256, 3))

    # test batch preprocessing of images from a directory
    def test_batch_preprocess(self):
        ims, filenames, failed = batch_imread(self.folder_path)
        im, fail = batch_preprocess(ims)
        total_face = len(im) + len(failed)
        self.assertEqual(total_face, len(os.listdir(self.folder_path)))

    # test batch writing preprocessed images to file
    def test_batch_imwrite(self):
        ims, filenames, failed = batch_imread(self.folder_path)
        im, fail = batch_preprocess(ims)
        output_path = './unit_test_samples/test_write'
        batch_imwrite(output_path, filenames, im)
        self.assertEqual(len(fail) + len(os.listdir(output_path)), len(ims))

    # test deppfake simulation
    def test_simulateDeepfake(self):
        ims, filenames, failed = batch_imread(self.folder_path)
        count = 0
        for i in range(len(ims)):
            im = cv2.imread(os.path.join(self.folder_path, filenames[i]))
            size = im.shape

            pixel1 = im[size[0]//2+5, size[1]//2+5]
            p1_1 = pixel1[0]
            p1_2 = pixel1[1]
            p1_3 = pixel1[2]

            faces = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)
            trans_matrix, point = faces[0]
            res = simulateDeepfake(im, trans_matrix, point)
            
            pixel2 = res[size[0]//2+5, size[1]//2+5]
            p2_1 = pixel2[0]
            p2_2 = pixel2[1]
            p2_3 = pixel2[2]

            if p1_1 != p2_1 or p1_2 != p2_2 or p2_1 != p2_3:
                count += 1
        self.assertEqual(count, len(ims))

    
    # test preprocess function used by prediction
    def test_prediction_preprocess(self):
        ims, filenames, failed = batch_imread(self.face_folder_path)
        for i in range(len(ims)):
            im = cv2.imread(os.path.join(self.face_folder_path, filenames[i]))
            face = prediction.preprocess(im)
            try:
                self.assertIsNotNone(face, "failed detecting face: "+ (filenames[i][:-4]).replace('_', ' '))
            except AssertionError as e:
                self.errors.append(str(e))

        ims, filenames, failed = batch_imread(self.without_face_folder_path)
        for i in range(len(ims)):
            im = cv2.imread(os.path.join(self.without_face_folder_path, filenames[i]))
            face = prediction.preprocess(im)
            try:
                self.assertIsNone(face)
            except AssertionError as e:
                self.errors.append(str(e))
    
    # test prediction function used on UI
    def test_prediction(self):
        ims, filenames, failed = batch_imread(self.folder_path)
        for i in range(len(ims)):
            img_path = os.path.join(self.folder_path, filenames[i])
            res = prediction.predict(img_path)
            self.assertIsInstance(res, bool)

        ims, filenames, failed = batch_imread(self.without_face_folder_path)
        for i in range(len(ims)):
            img_path = os.path.join(self.without_face_folder_path, filenames[i])
            res = prediction.predict(img_path)
            self.assertFalse(res)
            
        ims, filenames, failed = batch_imread(self.face_folder_video_path)
        for i in range(len(ims)):
            img_path = os.path.join(self.face_folder_video_path, filenames[i])
            res = prediction.predict(img_path)
            self.assertIsInstance(res, bool)

        ims, filenames, failed = batch_imread(self.without_face_folder_video_path)
        for i in range(len(ims)):
            img_path = os.path.join(self.without_face_folder_video_path, filenames[i])
            res = prediction.predict(img_path)
            self.assertFalse(res)




if __name__ == "__main__":
    unittest.main()







