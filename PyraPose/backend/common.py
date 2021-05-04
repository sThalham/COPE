import keras.backend
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


def box3D_transform_inv(boxes, deltas, mean=None, std=None):

    if mean is None:
        mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    boxes_exp = boxes

    width  = boxes_exp[:, :, 2] - boxes_exp[:, :, 0]
    height = boxes_exp[:, :, 3] - boxes_exp[:, :, 1]

    x1 = boxes_exp[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes_exp[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes_exp[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes_exp[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height
    x3 = boxes_exp[:, :, 0] + (deltas[:, :, 4] * std[4] + mean[4]) * width
    y3 = boxes_exp[:, :, 1] + (deltas[:, :, 5] * std[5] + mean[5]) * height
    x4 = boxes_exp[:, :, 2] + (deltas[:, :, 6] * std[6] + mean[6]) * width
    y4 = boxes_exp[:, :, 3] + (deltas[:, :, 7] * std[7] + mean[7]) * height
    x5 = boxes_exp[:, :, 0] + (deltas[:, :, 8] * std[8] + mean[8]) * width
    y5 = boxes_exp[:, :, 1] + (deltas[:, :, 9] * std[9] + mean[9]) * height
    x6 = boxes_exp[:, :, 2] + (deltas[:, :, 10] * std[10] + mean[10]) * width
    y6 = boxes_exp[:, :, 3] + (deltas[:, :, 11] * std[11] + mean[11]) * height
    x7 = boxes_exp[:, :, 0] + (deltas[:, :, 12] * std[12] + mean[12]) * width
    y7 = boxes_exp[:, :, 1] + (deltas[:, :, 13] * std[13] + mean[13]) * height
    x8 = boxes_exp[:, :, 2] + (deltas[:, :, 14] * std[14] + mean[14]) * width
    y8 = boxes_exp[:, :, 3] + (deltas[:, :, 15] * std[15] + mean[15]) * height

    pred_boxes = keras.backend.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8], axis=2)

    return pred_boxes


def box3D_denormalization(boxes, deltas, mean=None, std=None):

    if mean is None:
        mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    boxes_exp = boxes

    width  = boxes_exp[:, :, 2] - boxes_exp[:, :, 0]
    height = boxes_exp[:, :, 3] - boxes_exp[:, :, 1]

    x1 = (boxes_exp[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width) / 320.0
    y1 = (boxes_exp[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height) / 240.0
    x2 = (boxes_exp[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width) / 320.0
    y2 = (boxes_exp[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height) / 240.0
    x3 = (boxes_exp[:, :, 0] + (deltas[:, :, 4] * std[4] + mean[4]) * width) / 320.0
    y3 = (boxes_exp[:, :, 1] + (deltas[:, :, 5] * std[5] + mean[5]) * height) / 240.0
    x4 = (boxes_exp[:, :, 2] + (deltas[:, :, 6] * std[6] + mean[6]) * width) / 320.0
    y4 = (boxes_exp[:, :, 3] + (deltas[:, :, 7] * std[7] + mean[7]) * height) / 240.0
    x5 = (boxes_exp[:, :, 0] + (deltas[:, :, 8] * std[8] + mean[8]) * width) / 320.0
    y5 = (boxes_exp[:, :, 1] + (deltas[:, :, 9] * std[9] + mean[9]) * height) / 240.0
    x6 = (boxes_exp[:, :, 2] + (deltas[:, :, 10] * std[10] + mean[10]) * width) / 320.0
    y6 = (boxes_exp[:, :, 3] + (deltas[:, :, 11] * std[11] + mean[11]) * height) / 240.0
    x7 = (boxes_exp[:, :, 0] + (deltas[:, :, 12] * std[12] + mean[12]) * width) / 320.0
    y7 = (boxes_exp[:, :, 1] + (deltas[:, :, 13] * std[13] + mean[13]) * height) / 240.0
    x8 = (boxes_exp[:, :, 2] + (deltas[:, :, 14] * std[14] + mean[14]) * width) / 320.0
    y8 = (boxes_exp[:, :, 3] + (deltas[:, :, 15] * std[15] + mean[15]) * height) / 240.0

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

