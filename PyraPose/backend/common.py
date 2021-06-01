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


def box3D_transform_inv(regression, locations, mean=None, std=None):
    """
    regression: shape: (batch_size, location_shapes, 16+1)
    locations_shapes = P3_shape + P4_shape + P5_shape = (16+4+1)*P5_shape
    P3 shape: 60x80
    P4 shape: 30x40
    P5 shape: 15x20
    """
    # batch_size = regression[0].shape[0]
    # location_shapes = regression[0].shape[1]
    # level_shape = [[60, 80], [30, 40], [15, 20]]
    # level_scale = [8, 16, 32]
    #
    # import numpy as np
    #
    # locations = np.zeros((batch_size, location_shapes, 2))
    #
    # for batch in range(batch_size):
    #     counter = 0
    #     for level in range(3):
    #         for x in range(level_shape[level][0]):
    #             for y in range(level_shape[level][1]):
    #                 locations[batch, counter] = [x, y] * level_scale[level] + level_scale[level]/2
    #                 counter += 1

    if mean is None:
        mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if std is None:
        #std = [580, 580,  580,  580,  580,  580,  580,  580,  580,  580,  580,  580,  580,  580,  580,  580] #5200
        std = [750, 750,  750,  750,  750,  750,  750,  750,  750,  750,  750, 750,  750,  750,  750, 750] #5200

    x1 = locations[:, :, 0] - (regression[:, :, 0] * std[0] + mean[0])
    y1 = locations[:, :, 1] - (regression[:, :, 1] * std[1] + mean[1])
    x2 = locations[:, :, 0] - (regression[:, :, 2] * std[2] + mean[2])
    y2 = locations[:, :, 1] - (regression[:, :, 3] * std[3] + mean[3])
    x3 = locations[:, :, 0] - (regression[:, :, 4] * std[4] + mean[4])
    y3 = locations[:, :, 1] - (regression[:, :, 5] * std[5] + mean[5])
    x4 = locations[:, :, 0] - (regression[:, :, 6] * std[6] + mean[6])
    y4 = locations[:, :, 1] - (regression[:, :, 7] * std[7] + mean[7])
    x5 = locations[:, :, 0] - (regression[:, :, 8] * std[8] + mean[8])
    y5 = locations[:, :, 1] - (regression[:, :, 9] * std[9] + mean[9])
    x6 = locations[:, :, 0] - (regression[:, :, 10] * std[10] + mean[10])
    y6 = locations[:, :, 1] - (regression[:, :, 11] * std[11] + mean[11])
    x7 = locations[:, :, 0] - (regression[:, :, 12] * std[12] + mean[12])
    y7 = locations[:, :, 1] - (regression[:, :, 13] * std[13] + mean[13])
    x8 = locations[:, :, 0] - (regression[:, :, 14] * std[14] + mean[14])
    y8 = locations[:, :, 1] - (regression[:, :, 15] * std[15] + mean[15])

    pred_boxes = keras.backend.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8], axis=2)

    # h = regression[0].shape[0]
    # w = regression[0].shape[1]
    # x = np.ones((8, h, w)) * np.arange(w)
    # y = (np.ones((8, w, h)) * np.arange(h)).T
    # for i in range(0, 8):
    #     x[i] = x[i] - (regression[:, :, i*2] * std[i*2] + mean[i*2])
    #     y[i] = y[i] - (regression[:, :, i*2 + 1] * std[i*2 + 1] + mean[i*2 + 1])
    #
    # pred_boxes = keras.backend.stack([x[0], y[0], x[1], y[1], x[2], y[2], x[3], y[3], x[4], y[4], x[5], y[5], x[6], y[6], x[7], y[7]], axis=2)

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

