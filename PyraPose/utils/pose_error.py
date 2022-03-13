# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Implementation of the pose error functions described in:
# Hodan et al., "On Evaluation of 6D Object Pose Estimation", ECCVW 2016

import math
import numpy as np
from scipy import spatial
from transforms3d.quaternions import quat2mat, mat2quat
import cv2
#from .hodan_renderer import render


def estimate_visib_mask(d_test, d_model, delta):
    """
    Estimation of visibility mask.
    :param d_test: Distance image of the test scene.
    :param d_model: Rendered distance image of the object model.
    :param delta: Tolerance used in the visibility test.
    :return: Visibility mask.
    """
    assert(d_test.shape == d_model.shape)
    mask_valid = np.logical_and(d_test > 0, d_model > 0)

    d_diff = d_model.astype(np.float32) - d_test.astype(np.float32)
    visib_mask = np.logical_and(d_diff <= delta, mask_valid)

    return visib_mask


def estimate_visib_mask_gt(d_test, d_gt, delta):
    visib_gt = estimate_visib_mask(d_test, d_gt, delta)
    return visib_gt


def estimate_visib_mask_est(d_test, d_est, visib_gt, delta):
    visib_est = estimate_visib_mask(d_test, d_est, delta)
    visib_est = np.logical_or(visib_est, np.logical_and(visib_gt, d_est > 0))
    return visib_est


def depth_im_to_dist_im(depth_im, K):
    """
    Converts depth image to distance image.
    :param depth_im: Input depth image, where depth_im[y, x] is the Z coordinate
    of the 3D point [X, Y, Z] that projects to pixel [x, y], or 0 if there is
    no such 3D point (this is a typical output of the Kinect-like sensors).
    :param K: Camera matrix.
    :return: Distance image dist_im, where dist_im[y, x] is the distance from
    the camera center to the 3D point [X, Y, Z] that projects to pixel [x, y],
    or 0 if there is no such 3D point.
    """
    xs = np.tile(np.arange(depth_im.shape[1]), [depth_im.shape[0], 1])
    ys = np.tile(np.arange(depth_im.shape[0]), [depth_im.shape[1], 1]).T

    Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
    Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])

    dist_im = np.linalg.norm(np.dstack((Xs, Ys, depth_im)), axis=2)
    return dist_im


def transform_pts_Rt(pts, R, t):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert(pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def project2img(model, img_size, K, R, t):

    model = transform_pts_Rt(model, R, t)

    img = np.zeros(img_size, dtype=np.float32)
    mask = np.zeros(img_size, dtype=np.bool)
    xpix = ((model[:, 0] * K[0, 0]) / model[:, 2]) + K[0, 2]
    ypix = ((model[:, 1] * K[1, 1]) / model[:, 2]) + K[1, 2]

    for i in range(0, xpix.shape[0]):
        x = int(xpix[i])
        y = int(ypix[i])

        if y > img_size[0]-1 or x > img_size[1]-1 or y < 0 or x < 0:
            print(x)
            print(y)
            continue

        if not mask[y, x]:
            img[y, x] = model[i,2] * 1000.0
            mask[y, x] = True
        elif (model[i, 2]*1000.0) < img[y, x]:
            img[y, x] = model[i, 2]*1000.0

    return img


def vsd(R_est, t_est, R_gt, t_gt, model, depth_test, K, delta, tau,
        cost_type='step'):
    """
    Visible Surface Discrepancy.
    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :param depth_test: Depth image of the test scene.
    :param K: Camera matrix.
    :param delta: Tolerance used for estimation of the visibility masks.
    :param tau: Misalignment tolerance.
    :param cost_type: Pixel-wise matching cost:
        'tlinear' - Used in the original definition of VSD in:
            Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW 2016
        'step' - Used for SIXD Challenge 2017. It is easier to interpret.
    :return: Error of pose_est w.r.t. pose_gt.
    """

    im_size = (depth_test.shape[1], depth_test.shape[0])

    #depth_est = project2img(model, im_size, K, R_est, t_est)

    #depth_gt = project2img(model, im_size, K, R_gt, t_gt)

    depth_est = render(model, im_size, K, R_est, t_est, clip_near=100,
                                clip_far=10000, mode='depth')

    depth_gt = render(model, im_size, K, R_gt, t_gt, clip_near=100,
                               clip_far=10000, mode='depth')

    # Convert depth images to distance images
    dist_test = depth_im_to_dist_im(depth_test, K)
    dist_gt = depth_im_to_dist_im(depth_gt, K)
    dist_est = depth_im_to_dist_im(depth_est, K)

    # Visibility mask of the model in the ground truth pose
    visib_gt = estimate_visib_mask_gt(dist_test, dist_gt, delta)


    # Visibility mask of the model in the estimated pose
    visib_est = estimate_visib_mask_est(dist_test, dist_est, visib_gt, delta)

    #scaCro = 255.0 / np.nanmax(depth_est)
    #cross = np.multiply(depth_est, scaCro)
    #dep_sca = cross.astype(np.uint8)
    #cv2.imwrite("/home/sthalham/visTests/depth_est.jpg", dep_sca)

    # Intersection and union of the visibility masks
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)

    # Pixel-wise matching cost
    costs = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
    if cost_type == 'step':
        costs = costs >= tau
    elif cost_type == 'tlinear': # Truncated linear
        costs *= (1.0 / tau)
        costs[costs > 1.0] = 1.0
    else:
        print('Error: Unknown pixel matching cost.')
        exit(-1)

    # Visible Surface Discrepancy
    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()
    if visib_union_count > 0:
        e = (costs.sum() + visib_comp_count) / float(visib_union_count)
    else:
        e = 1.0

    return e


def reproj(K, R_est, t_est, R_gt, t_gt, pts):
    """
    reprojection error.
    :param K intrinsic matrix
    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    pixels_est = K.dot(pts_est.T)
    pixels_est = pixels_est.T
    pixels_gt = K.dot(pts_gt.T)
    pixels_gt = pixels_gt.T

    n = pts.shape[0]
    est = np.zeros((n, 2), dtype=np.float32)
    est[:, 0] = np.divide(pixels_est[:, 0], pixels_est[:, 2])
    est[:, 1] = np.divide(pixels_est[:, 1], pixels_est[:, 2])

    gt = np.zeros((n, 2), dtype=np.float32)
    gt[:, 0] = np.divide(pixels_gt[:, 0], pixels_gt[:, 2])
    gt[:, 1] = np.divide(pixels_gt[:, 1], pixels_gt[:, 2])

    e = np.linalg.norm(est - gt, axis=1).mean()
    return e


def add(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e


def adi(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from pts_gt to pts_est
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e


def re(R_est, R_gt):
    """
    Rotational Error.

    :param R_est: Rotational element of the estimated pose (3x1 vector).
    :param R_gt: Rotational element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(R_est.shape == R_gt.shape == (3, 3))
    error_cos = 0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0)
    error_cos = min(1.0, max(-1.0, error_cos)) # Avoid invalid values due to numerical errors
    error = math.acos(error_cos)
    error = 180.0 * error / np.pi # [rad] -> [deg]
    return error


def te(t_est, t_gt):
    """
    Translational Error.

    :param t_est: Translation element of the estimated pose (3x1 vector).
    :param t_gt: Translation element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt - t_est)
    return error
