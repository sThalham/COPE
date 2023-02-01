
from __future__ import division
import numpy as np
import cv2
import math
import transforms3d as tf3d
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .transform import change_transform_origin


def read_image_bgr(path):
    """ Read an image in BGR format.

    Args
        path: Path to the image.
    """

    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()


def read_image_dep(path):
    """ Read an image in BGR format.

    Args
        path: Path to the image.
    """
    image = np.asarray(Image.open(path))
    return image[:, :].copy()


def preprocess_image(x, mode='caffe'):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


def adjust_transform_for_mask(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width = image.shape

    result = transform.copy()

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    def __init__(
        self,
        fill_mode            = 'nearest',
        interpolation        = 'linear',
        cval                 = 0,
        relative_translation = True,
    ):
        self.fill_mode            = fill_mode
        self.cval                 = cval
        self.interpolation        = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params):

    image = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize       = (image.shape[1], image.shape[0]),
        flags       = params.cvInterpolation(),
        borderMode  = params.cvBorderMode(),
        borderValue = params.cval,
    )

    return image


def apply_transform2mask(matrix, mask, params, min_side=480, max_side=640):

    mask = np.asarray(Image.fromarray(mask).resize((max_side, min_side), Image.NEAREST))
    mask = cv2.warpAffine(
        mask,
        matrix[:2, :],
        dsize=(mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return [mask]


def augment_image(image, sequential):
    # rgb
    # seq describes an object for rgb image augmentation using aleju/imgaug

    image = sequential.augment_image(image)

    return image


def adjust_pose_annotation(matrix, pose, cpara, image_shape):

    # x = (pix * z) / f
    translation_x = matrix[0][0, 2]
    translation_y = matrix[0][1, 2]
    scale = matrix[1][0, 0]
    rotation = np.arccos(matrix[2][0,0])

    # scale
    cpara[0] = cpara[0] * scale
    cpara[1] = cpara[1] * scale
    # translate
    cpara[2] = cpara[2] + (translation_x * image_shape[1])
    cpara[3] = cpara[3] + (translation_y * image_shape[0])

    #trans_cam = np.eye(4)
    #trans_pose = np.eye(4)
    #trans_cam[:3, :3] = tf3d.euler.euler2mat(rotation, 0.0, 0.0)#, 'sxyz')
    #trans_pose[:3, :3] = tf3d.quaternions.quat2mat(pose[3:])
    #trans_pose[:3, 3] = pose[:3]
    #print('rotation: ', rot_cam)
    #print('pose:  ', rot_pose)
    #aug_trans = np.linalg.inv(np.linalg.inv(trans_pose) @ trans_cam)
    #aug_trans = np.linalg.inv(np.linalg.inv(trans_cam) @ trans_pose)
    #pose[3:] = tf3d.quaternions.mat2quat(aug_trans[:3, :3])
    #pose[:3] = aug_trans[:3, 3]

    #print(matrix)

    #cpara[0] = cpara[0] * scale
    #cpara[1] = cpara[1] * scale
    #cpara[2] = cpara[2] + matrix[0, 2] # * scale
    #cpara[3] = cpara[3] + matrix[1, 2] # * scale
    #cpara[2] = matrix[0, 2]
    #cpara[3] = matrix[1, 2]

    #pose[2] = pose[2] / scale
    #pose[0] = pose[0] + ((matrix[0, 2] + ((cpara[2] * matrix[0, 0]) - cpara[2])) * pose[2]) / cpara[0]
    #pose[1] = pose[1] + ((matrix[1, 2] + ((cpara[3] * matrix[0, 0]) - cpara[3])) * pose[2]) / cpara[1]

    #trans_aug = np.array([pose[0], pose[1], pose[2]])
    #R_2naug = lookAt(trans_noaug, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    #R_2aug = lookAt(trans_aug, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    #R_rel = np.linalg.inv(R_2naug[:3, :3]) @ R_2aug[:3, :3]
    #R_aug = R_rel @ tf3d.quaternions.quat2mat(pose[3:7])
    #R_rel = np.linalg.inv(R_2aug[:3, :3]) @ R_2naug[:3, :3]
    #R_aug = np.linalg.inv(np.linalg.inv(tf3d.quaternions.quat2mat(pose[3:7])) @ R_rel)
    #pose[3:] = tf3d.quaternions.mat2quat(R_aug)

    return pose, cpara


def compute_resize_scale(image_shape, min_side=480, max_side=640):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=480, max_side=640):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale
