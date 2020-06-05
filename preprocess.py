import random, cv2, os, dlib, numpy as np
from py_utils.face_utils import lib

'''
Carries out preprocessing on a batch of images provided. Note in its current state, only works on images, not videos.
Preprocessing for positive samples includes only resizing.
For negative samples, the resolution inconsistency of Deepfake is simulated using Gaussian blurring and affine warping.
'''

pwd = os.path.dirname(__file__)

# minimum and maximum range for facial region to be resized to (pixels x pixels), before being blurred and affine warped back to the original image
RESIZE_MIN = 64
RESIZE_MAX = 128

# output image size = expected input size to network architecture
IMG_SIZE = 256

FOLDER_PATH_POSITIVE_IMGS = pwd + "./imgs"
FOLDER_PATH_NEGATIVE_IMGS = pwd + "./imgs" # this is the folder where the images will be applied Deepfakes simulation

front_face_detector = dlib.get_frontal_face_detector()
lmark_predictor = dlib.shape_predictor(pwd + './dlib_model/shape_predictor_68_face_landmarks.dat')


def simulateDeepfake(im, trans_matrix, point):

    '''
    Given an image, a transform matrix and landmark points of a detected facial region, applies appropriate
    Gaussian blurring and affine warping to simulate the resolution inconsistencies of Deepfake.
    Masking is used to only blur the region shaped as the face, instead of a rectangular box around the face.
    Returns the simulated image.
    '''

    image_size = im.shape[1], im.shape[0]
    size = random.randint(RESIZE_MIN, RESIZE_MAX + 1) # align face to a size k x k where k is a number between RESIZE_MIN and RESIZE_MAX

    # Figure 2 in CVPRW2019_Face_Artifacts

    face = cv2.warpAffine(im, trans_matrix * size, (size, size))
    face = cv2.GaussianBlur(face, (5,5), 0)

    warped_im = np.copy(im)
    warped_im = cv2.warpAffine(face, trans_matrix*size, image_size, warped_im, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT)

    # Figure 3 in CVPRW2019_Face_Artifacts

    # get mask of facial area and surrounds, convert to grayscale
    mask = lib.get_face_mask(im.shape[:2], point)
    mask_inv = cv2.bitwise_not(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_BGR2GRAY)

    # masking and combining output
    area_surrounding = cv2.bitwise_or(im, im, mask = mask_inv)
    area_surrounding = area_surrounding[0:im.shape[0], 0:im.shape[1]]

    area_facial = cv2.bitwise_or(warped_im, warped_im, mask = mask)
    area_facial = area_facial[0:im.shape[0], 0:im.shape[1]]

    output_im = area_surrounding + area_facial

    return output_im


def preprocess(im, willSimulateDeepfake=False):

    '''
    Given an image, attempts to detect a face. If none is found, returns None. Returns the image resized to as specified by IMG_SIZE.
    If instructed to simulate Deepfake, simulation will be applied before resizing.
    '''

    # list of tuples of (transformation matrix, landmark point) of identified faces
    faces = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)
    
    if len(faces) == 0:
        return None

    trans_matrix, point = faces[0] # take only the first face found

    if willSimulateDeepfake:
        im = simulateDeepfake(im, trans_matrix, point)

    # GENERALIZATION STEP. REMOVED AFTER ACCURACY LOWERED.
    # crop image, after randomly expanding ROI (minimum bounding box b) in each direction between
    # [0, h/5] and [0, w/8] where h, w are height and width of b. then resize to 256 x 256 for final training data
    # rois, _ = lib.cut_head([im], point, random.randint(0, 10))
    # cropped_output_im = cv2.resize(rois[0], (IMG_SIZE, IMG_SIZE))

    im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))

    return im


def batch_preprocess(ims, willSimulateDeepfake=False):

    '''
    Given an array of images, preprocess them in batch and returns the array of successfully preprocessed images.
    Failed ones will be discarded, and their indicies wil be returned.
    Will apply simulation as appropriate if instructed.
    '''

    output_ims = []
    failed_indices = []

    for i in range(len(ims)):
        output = preprocess(ims[i], willSimulateDeepfake)
        if output is not None:
            output_ims.append(output)
        else: # no face found
            failed_indices.append(i)

    return output_ims, failed_indices


def batch_imread(folder_path):

    '''
    Given a folder path containing images, read them in batch and returns the array of successfully read images. (failed ones discarded)
    Also returns the filenames as well as the indices of files that have failed to be read.
    '''

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

    '''
    Given a folder path, a list of filenames for every image, and an array of images, write the images out to the path + filename.
    Also adds a suffix of "_preprocessed" before the filetype.
    '''

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in range(len(ims)):
        new_filename = filenames[i][:-4] + "_preprocessed" + filenames[i][-4:]
        path = os.path.join(folder_path, new_filename)
        cv2.imwrite(path, ims[i])


def readAndPreprocessAndWrite(path, outPath, willSimulateDeepfake=False):

    '''
    Given a path containing images, an output path, and a flag of whether to simulate Deepfake effects,
    preprocess the images and write out to the specified output path.
    '''

    print("Reading all images from directory...")

    ims, filenames, failed = batch_imread(path)

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

    output_ims, failed = batch_preprocess(ims, willSimulateDeepfake)

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

    batch_imwrite(outPath, filenames, output_ims)

    print("OK")
    print("Done.")


if __name__ == "__main__":

    # original, non-simulated
    print("Performing preprocessing actions on positive training images...")
    readAndPreprocessAndWrite(FOLDER_PATH_POSITIVE_IMGS, os.path.join(FOLDER_PATH_POSITIVE_IMGS, "./pos"), False)

    # Deepfakes resolution inconsistencies simulated
    print("Performing preprocessing actions on negative training images...")
    readAndPreprocessAndWrite(FOLDER_PATH_NEGATIVE_IMGS, os.path.join(FOLDER_PATH_NEGATIVE_IMGS, "./neg"), True)
