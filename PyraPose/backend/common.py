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


def box3D_transform_inv(regression, locations, obj_diameter, mean=None, std=None):

    if mean is None:
        mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if std is None:
        std = [0.65, 0.65,  0.65,  0.65,  0.65,  0.65,  0.65,  0.65,  0.65,  0.65,  0.65, 0.65, 0.65, 0.65, 0.65, 0.65]

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


def box3D_denorm(regression, locations, mean=None, std=None):

    if mean is None:
        mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if std is None:
        std = [0.65, 0.65,  0.65,  0.65,  0.65,  0.65,  0.65,  0.65,  0.65,  0.65,  0.65, 0.65, 0.65, 0.65, 0.65, 0.65]

    #tf.print('locations: ', tf.shape(locations))
    #tf.print('regression: ', tf.shape(regression))
    #tf.print('obj_diameter: ', tf.shape(obj_diameter))

    x1 = locations[:, :, :, 0] - (regression[:, :, :, 0] * std[0]  + mean[0])
    y1 = locations[:, :, :, 1] - (regression[:, :, :, 1] * std[1]  + mean[1])
    x2 = locations[:, :, :, 0] - (regression[:, :, :, 2] * std[2]  + mean[2])
    y2 = locations[:, :, :, 1] - (regression[:, :, :, 3] * std[3]  + mean[3])
    x3 = locations[:, :, :, 0] - (regression[:, :, :, 4] * std[4]  + mean[4])
    y3 = locations[:, :, :, 1] - (regression[:, :, :, 5] * std[5]  + mean[5])
    x4 = locations[:, :, :, 0] - (regression[:, :, :, 6] * std[6]  + mean[6])
    y4 = locations[:, :, :, 1] - (regression[:, :, :, 7] * std[7]  + mean[7])
    x5 = locations[:, :, :, 0] - (regression[:, :, :, 8] * std[8]  + mean[8])
    y5 = locations[:, :, :, 1] - (regression[:, :, :, 9] * std[9]  + mean[9])
    x6 = locations[:, :, :, 0] - (regression[:, :, :, 10] * std[10] + mean[10])
    y6 = locations[:, :, :, 1] - (regression[:, :, :, 11] * std[11] + mean[11])
    x7 = locations[:, :, :, 0] - (regression[:, :, :, 12] * std[12] + mean[12])
    y7 = locations[:, :, :, 1] - (regression[:, :, :, 13] * std[13] + mean[13])
    x8 = locations[:, :, :, 0] - (regression[:, :, :, 14] * std[14] + mean[14])
    y8 = locations[:, :, :, 1] - (regression[:, :, :, 15] * std[15] + mean[15])

    #x1 = locations[:, :, :, 0] - (regression[:, :, :, 0] * (std[0] * obj_diameter) + mean[0])
    #y1 = locations[:, :, :, 1] - (regression[:, :, :, 1] * (std[1] * obj_diameter) + mean[1])
    #x2 = locations[:, :, :, 0] - (regression[:, :, :, 2] * (std[2] * obj_diameter) + mean[2])
    #y2 = locations[:, :, :, 1] - (regression[:, :, :, 3] * (std[3] * obj_diameter) + mean[3])
    #x3 = locations[:, :, :, 0] - (regression[:, :, :, 4] * (std[4] * obj_diameter) + mean[4])
    #y3 = locations[:, :, :, 1] - (regression[:, :, :, 5] * (std[5] * obj_diameter) + mean[5])
    #x4 = locations[:, :, :, 0] - (regression[:, :, :, 6] * (std[6] * obj_diameter) + mean[6])
    #y4 = locations[:, :, :, 1] - (regression[:, :, :, 7] * (std[7] * obj_diameter) + mean[7])
    #x5 = locations[:, :, :, 0] - (regression[:, :, :, 8] * (std[8] * obj_diameter) + mean[8])
    #y5 = locations[:, :, :, 1] - (regression[:, :, :, 9] * (std[9] * obj_diameter) + mean[9])
    #x6 = locations[:, :, :, 0] - (regression[:, :, :, 10] * (std[10] * obj_diameter) + mean[10])
    #y6 = locations[:, :, :, 1] - (regression[:, :, :, 11] * (std[11] * obj_diameter) + mean[11])
    #x7 = locations[:, :, :, 0] - (regression[:, :, :, 12] * (std[12] * obj_diameter) + mean[12])
    #y7 = locations[:, :, :, 1] - (regression[:, :, :, 13] * (std[13] * obj_diameter) + mean[13])
    #x8 = locations[:, :, :, 0] - (regression[:, :, :, 14] * (std[14] * obj_diameter) + mean[14])
    #y8 = locations[:, :, :, 1] - (regression[:, :, :, 15] * (std[15] * obj_diameter) + mean[15])

    pred_boxes = keras.backend.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8], axis=3)

    return pred_boxes


def poses_denorm(regression):

    # quaternions and translation
    #x = regression[:, :, :, 0] * 500.0
    #y = regression[:, :, :, 1] * 500.0
    #z = ((regression[:, :, :, 2] * (1/3)) + 1.0) * 1000.0
    #q = regression[:, :, :, 3]
    #r = regression[:, :, :, 4]
    #p = regression[:, :, :, 5]
    #g = regression[:, :, :, 6]
    #pred_poses = keras.backend.stack([x, y, z, q, r, p, g], axis=3)

    # dual quaternions
    #q = regression[:, :, :, 0]
    #r = regression[:, :, :, 1]
    #p = regression[:, :, :, 2]
    #g = regression[:, :, :, 3]
    #s = regression[:, :, :, 4] / 0.003
    #x = regression[:, :, :, 5] / 0.002
    #y = regression[:, :, :, 6] / 0.002
    #z = regression[:, :, :, 7] / 0.003
    #pred_poses = keras.backend.stack([q, r, p, g, s, x, y, z], axis=3)

    # r6d
    x = regression[:, :, :, 0] * 500.0
    y = regression[:, :, :, 1] * 500.0
    z = ((regression[:, :, :, 2] * (1/3)) + 1.0) * 1000.0
    r00 = regression[:, :, :, 3]
    r01 = regression[:, :, :, 4]
    r02 = regression[:, :, :, 5]
    r10 = regression[:, :, :, 6]
    r11 = regression[:, :, :, 7]
    r12 = regression[:, :, :, 8]
    pred_poses = keras.backend.stack([x, y, z, r00, r01, r02, r10, r11, r12], axis=3)

    return pred_poses


def box_projection(poses, corres, intrinsics):

    # todo
    #rot = tf3d.quaternions.quat2mat(pose[3:])
    #rot = np.asarray(rot, dtype=np.float32)
    #tra = pose[:3]
    #tDbox = rot[:3, :3].dot(annotations['segmentations'][idx].T).T
    #tDbox = tDbox + np.repeat(tra[np.newaxis, 0:3], 8, axis=0)

    #box3D = toPix_array(tDbox, fx=annotations['cam_params'][idx][0], fy=annotations['cam_params'][idx][1],
    #                    cx=annotations['cam_params'][idx][2], cy=annotations['cam_params'][idx][3])

    #xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    #ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy

    # r6d
    x = poses[:, :, :, 0] * 500.0
    y = poses[:, :, :, 1] * 500.0
    z = ((poses[:, :, :, 2] * (1 / 3)) + 1.0) * 1000.0
    trans = tf.stack([x, y, z], axis=3)
    trans = tf.tile(trans[:, :, :, tf.newaxis, :], [1, 1, 1, 8, 1])

    #r1 = tf.stack([poses[:, :, :, 3], poses[:, :, :, 4], poses[:, :, :, 5]], axis=3)
    #r1 = tf.math.l2_normalize(poses[:, :, :, 3:6], axis=3)
    r1 = poses[:, :, :, 3:6]
    #r2 = tf.stack([regression[:, :, :, 6], regression[:, :, :, 7], regression[:, :, :, 8]], axis=3)
    #r2 = tf.math.l2_normalize(poses[:, :, :, 6:], axis=3)
    r2 = poses[:, :, :, 6:]
    r3 = tf.linalg.cross(r1, r2)
    r3 = tf.math.l2_normalize(r3, axis=3)
    rot = tf.stack([r1, r2, r3], axis=4)

    print('rot: ', rot)
    print('corres: ', corres)
    print('trans: ', trans)

    #box3d = tf.tensordot(rot, corres, axes=[3, 4])
    box3d = tf.einsum('blckv, blcpk->blcpk', rot, corres)
    tf.print('box3d: ', box3d[0, 0, 0, :, :])
    box3d = tf.math.add(box3d, trans)

    projected_boxes_x = box3d[:, :, :, :, 0] * intrinsics[0]
    projected_boxes_x = tf.math.divide_no_nan(projected_boxes_x, box3d[:, :, :, :, 2])
    projected_boxes_x = tf.math.add(projected_boxes_x, intrinsics[1])
    projected_boxes_y = box3d[:, :, :, :, 1] * intrinsics[2]
    projected_boxes_y = tf.math.divide_no_nan(projected_boxes_y, box3d[:, :, :, :, 2])
    projected_boxes_y = tf.math.add(projected_boxes_y, intrinsics[3])
    pro_boxes = tf.stack([projected_boxes_x, projected_boxes_y], axis=3)
    tf.print(tf.shape(corres)[:2])
    pro_boxes = tf.reshape(pro_boxes, shape=[tf.shape(poses)[0], tf.shape(poses)[1], tf.shape(poses)[2], 16])

    return pro_boxes


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

