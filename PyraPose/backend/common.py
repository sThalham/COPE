import tensorflow.keras as keras
import tensorflow as tf
from tensorflow import meshgrid


def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width  = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = keras.backend.stack([x1, y1, x2, y2], axis=2)

    return pred_boxes


def box3D_transform_inv(regression, locations, labels, mean=None, std=None, diameters=None):

    if mean is None:
        mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if std is None:
        std = [0.65, 0.65,  0.65,  0.65,  0.65,  0.65,  0.65,  0.65,  0.65,  0.65,  0.65, 0.65, 0.65, 0.65, 0.65, 0.65]

    for idx in labels:

    #obj_diameter = diameters[labels] * 1000.0

    x1 = locations[:, :, 0] - (regression[:, :, 0] * (std[0] * obj_diameter) + mean[0])
    y1 = locations[:, :, 1] - (regression[:, :, 1] * (std[1] * obj_diameter) + mean[1])
    x2 = locations[:, :, 0] - (regression[:, :, 2] * (std[2] * obj_diameter) + mean[2])
    y2 = locations[:, :, 1] - (regression[:, :, 3] * (std[3] * obj_diameter) + mean[3])
    x3 = locations[:, :, 0] - (regression[:, :, 4] * (std[4] * obj_diameter) + mean[4])
    y3 = locations[:, :, 1] - (regression[:, :, 5] * (std[5] * obj_diameter) + mean[5])
    x4 = locations[:, :, 0] - (regression[:, :, 6] * (std[6] * obj_diameter) + mean[6])
    y4 = locations[:, :, 1] - (regression[:, :, 7] * (std[7] * obj_diameter) + mean[7])
    x5 = locations[:, :, 0] - (regression[:, :, 8] * (std[8] * obj_diameter) + mean[8])
    y5 = locations[:, :, 1] - (regression[:, :, 9] * (std[9] * obj_diameter) + mean[9])
    x6 = locations[:, :, 0] - (regression[:, :, 10] * (std[10] * obj_diameter) + mean[10])
    y6 = locations[:, :, 1] - (regression[:, :, 11] * (std[11] * obj_diameter) + mean[11])
    x7 = locations[:, :, 0] - (regression[:, :, 12] * (std[12] * obj_diameter) + mean[12])
    y7 = locations[:, :, 1] - (regression[:, :, 13] * (std[13] * obj_diameter) + mean[13])
    x8 = locations[:, :, 0] - (regression[:, :, 14] * (std[14] * obj_diameter) + mean[14])
    y8 = locations[:, :, 1] - (regression[:, :, 15] * (std[15] * obj_diameter) + mean[15])

    pred_boxes = keras.backend.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8], axis=2)

    return pred_boxes


def shift(shape, stride, anchors):
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors

