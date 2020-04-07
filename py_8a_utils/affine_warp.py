import numpy as np

# ref 1: https://medium.com/hipster-color-science/computing-2d-affine-transformations-using-only-matrix-multiplication-2ccb31b52181
# ref 2: https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20warpAffine(InputArray%20src,%20OutputArray%20dst,%20InputArray%20M,%20Size%20dsize,%20int%20flags,%20int%20borderMode,%20const%20Scalar&%20borderValue)

def affineWarp(im, trans_mat, outputSize, output, inv=False):

    # inverse affine trans. matrix if inverse affine warp
    if inv: # ref 3: https://stackoverflow.com/questions/2624422/efficient-4x4-matrix-inverse-affine-transform
        topleft = trans_mat[0:2, 0:2]
        topleft_inv = np.linalg.inv(topleft)
        topright = trans_mat[0:2, 1:2]
        topright_inv = np.negative(topleft_inv).dot(topright)
        trans_mat = np.hstack((topleft_inv, topright_inv))

    # turning 2 x 3 matrix to 3 x 3 augmented matrix (see ref 1)
    new_row = np.array([0, 0, 1])
    trans_mat = np.vstack([trans_mat, new_row])

    # for every pixel in source img, lookup newly-transformed pixel coords and transfer pixel color data
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            coord_mat = np.array([[x], [y], [1]]) # turn 2 x 1 mat to 3 x 1 augmented matrix (see ref 1)
            new_coord_mat = trans_mat.dot(coord_mat)
            # if int(new_coord_mat[0]) >= 0 and int(new_coord_mat[0]) <= outputSize[0] and int(new_coord_mat[1]) >= 0 and int(new_coord_mat[1]) <= outputSize[1]:
            #     print("yay, it's", new_coord_mat[0], new_coord_mat[1])
            try: # cut off if out-of-bounds
                output[int(new_coord_mat[0])][int(new_coord_mat[1])] = im[x][y]
            except IndexError:
                # print("out of index", new_coord_mat[0], new_coord_mat[1])
                continue

    return output