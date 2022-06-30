"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow.keras as keras
import tensorflow as tf
from . import backend


def focal(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha. vanilla 0.25 2.0
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def instanced_focal(arg):
    alpha = 0.25
    gamma = 2.0

    y_true, y_pred = arg

    labels = y_true[:, :, :-1]
    anchor_state = y_true[:, :, -1]
    classification = y_pred

    indices = backend.where(keras.backend.not_equal(anchor_state, -1))
    labels = backend.gather_nd(labels, indices)
    classification = backend.gather_nd(classification, indices)

    # compute the focal loss
    alpha_factor = keras.backend.ones_like(labels) * alpha
    alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
    focal_weight = alpha_factor * focal_weight ** gamma

    cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

    normalizer = backend.where(keras.backend.equal(anchor_state, 1))
    normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
    normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)
    #loss = tf.math.reduce_sum(cls_loss) / normalizer
    loss = cls_loss / normalizer

    return (loss, loss)


def instanced_cross(arg):
    sigma_squared = 9.0

    y_true, y_pred = arg

    labels = y_true[:, :, :-1]
    anchor_state = y_true[:, :, -1]
    classification = y_pred

    #indices = backend.where(keras.backend.equal(anchor_state, 1))
    indices = backend.where(keras.backend.not_equal(anchor_state, -1))
    labels = backend.gather_nd(labels, indices)
    classification = backend.gather_nd(classification, indices)

    cls_loss = keras.losses.binary_crossentropy(labels, classification)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.math.maximum(1, tf.shape(indices)[0])
    normalizer = tf.cast(normalizer, dtype=tf.float32)
    loss = tf.math.reduce_sum(cls_loss) / normalizer
    loss = cls_loss / normalizer

    return (loss, loss)


def per_cls_cross(num_classes=0, weight=1.0):

    def _per_cls_l1(y_true, y_pred):
        #y_true_exp = tf.expand_dims(y_true, axis=0)
        #y_true_rep = tf.tile(y_true_exp, [1, 1, num_classes, 1])
        y_true_perm = tf.transpose(y_true, [2, 0, 1, 3])

        y_pred_exp = tf.expand_dims(y_pred, axis=0)
        y_pred_rep = tf.tile(y_pred_exp, [num_classes, 1, 1, 1])

        loss_per_cls = tf.map_fn(instanced_cross, (y_true_perm, y_pred_rep))
        #loss_per_cls = tf.map_fn(instanced_focal, (y_true_perm, y_pred_rep))

        return weight * (tf.math.reduce_sum(loss_per_cls) / num_classes)

    return _per_cls_l1



def smooth_l1(weight=1.0, sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(regression_loss) / normalizer

        return weight * loss

    return _smooth_l1


def weighted_msle(weight=5.0):

    def _msle(y_true, y_pred):

        regression        = y_pred
        regression_target = y_true[:, :, :, :-1]
        anchor_state      = y_true[:, :, :, -1]

        # somethings fucky here
        #### filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        regression_loss = weight * keras.losses.mean_squared_logarithmic_error(regression, regression_target)

        #### compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return regression_loss / normalizer

    return _msle


def orthogonal_l1(weight=1.0, sigma=3.0):

    weight_xy = 0.8
    weight_orth = 0.2
    sigma_squared = sigma ** 2

    def _orth_l1(y_true, y_pred):

        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        #### filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        x1 = (regression[:, 0] - regression[:, 6]) - (regression[:, 2] - regression[:, 4])
        y1 = (regression[:, 1] - regression[:, 7]) - (regression[:, 3] - regression[:, 5])
        x2 = (regression[:, 0] - regression[:, 6]) - (regression[:, 8] - regression[:, 14])
        y2 = (regression[:, 1] - regression[:, 7]) - (regression[:, 9] - regression[:, 15])
        x3 = (regression[:, 0] - regression[:, 2]) - (regression[:, 6] - regression[:, 4])
        y3 = (regression[:, 1] - regression[:, 3]) - (regression[:, 7] - regression[:, 5])
        x4 = (regression[:, 0] - regression[:, 2]) - (regression[:, 8] - regression[:, 10])
        y4 = (regression[:, 1] - regression[:, 3]) - (regression[:, 9] - regression[:, 11])   # up to here ok
        x5 = (regression[:, 0] - regression[:, 8]) - (regression[:, 2] - regression[:, 10])
        y5 = (regression[:, 1] - regression[:, 9]) - (regression[:, 3] - regression[:, 11])
        x6 = (regression[:, 0] - regression[:, 8]) - (regression[:, 6] - regression[:, 14])
        y6 = (regression[:, 1] - regression[:, 9]) - (regression[:, 7] - regression[:, 15])   # half way done
        x7 = (regression[:, 12] - regression[:, 10]) - (regression[:, 14] - regression[:, 8])
        y7 = (regression[:, 13] - regression[:, 11]) - (regression[:, 15] - regression[:, 9])
        x8 = (regression[:, 12] - regression[:, 10]) - (regression[:, 4] - regression[:, 2])
        y8 = (regression[:, 13] - regression[:, 11]) - (regression[:, 5] - regression[:, 3])
        x9 = (regression[:, 12] - regression[:, 4]) - (regression[:, 10] - regression[:, 2])
        y9 = (regression[:, 13] - regression[:, 5]) - (regression[:, 11] - regression[:, 3])
        x10 = (regression[:, 12] - regression[:, 4]) - (regression[:, 14] - regression[:, 6])
        y10 = (regression[:, 13] - regression[:, 5]) - (regression[:, 15] - regression[:, 7])
        x11 = (regression[:, 12] - regression[:, 14]) - (regression[:, 4] - regression[:, 6])
        y11 = (regression[:, 13] - regression[:, 15]) - (regression[:, 5] - regression[:, 7])
        x12 = (regression[:, 12] - regression[:, 14]) - (regression[:, 10] - regression[:, 8])
        y12 = (regression[:, 13] - regression[:, 15]) - (regression[:, 11] - regression[:, 9])
        orths = keras.backend.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12], axis=1)

        xt1 = (regression_target[:, 0] - regression_target[:, 6]) - (regression_target[:, 2] - regression_target[:, 4])
        yt1 = (regression_target[:, 1] - regression_target[:, 7]) - (regression_target[:, 3] - regression_target[:, 5])
        xt2 = (regression_target[:, 0] - regression_target[:, 6]) - (regression_target[:, 8] - regression_target[:, 14])
        yt2 = (regression_target[:, 1] - regression_target[:, 7]) - (regression_target[:, 9] - regression_target[:, 15])
        xt3 = (regression_target[:, 0] - regression_target[:, 2]) - (regression_target[:, 6] - regression_target[:, 4])
        yt3 = (regression_target[:, 1] - regression_target[:, 3]) - (regression_target[:, 7] - regression_target[:, 5])
        xt4 = (regression_target[:, 0] - regression_target[:, 2]) - (regression_target[:, 8] - regression_target[:, 10])
        yt4 = (regression_target[:, 1] - regression_target[:, 3]) - (regression_target[:, 9] - regression_target[:, 11])  # up to here ok
        xt5 = (regression_target[:, 0] - regression_target[:, 8]) - (regression_target[:, 2] - regression_target[:, 10])
        yt5 = (regression_target[:, 1] - regression_target[:, 9]) - (regression_target[:, 3] - regression_target[:, 11])
        xt6 = (regression_target[:, 0] - regression_target[:, 8]) - (regression_target[:, 6] - regression_target[:, 14])
        yt6 = (regression_target[:, 1] - regression_target[:, 9]) - (regression_target[:, 7] - regression_target[:, 15])  # half way done
        xt7 = (regression_target[:, 12] - regression_target[:, 10]) - (regression_target[:, 14] - regression_target[:, 8])
        yt7 = (regression_target[:, 13] - regression_target[:, 11]) - (regression_target[:, 15] - regression_target[:, 9])
        xt8 = (regression_target[:, 12] - regression_target[:, 10]) - (regression_target[:, 4] - regression_target[:, 2])
        yt8 = (regression_target[:, 13] - regression_target[:, 11]) - (regression_target[:, 5] - regression_target[:, 3])
        xt9 = (regression_target[:, 12] - regression_target[:, 4]) - (regression_target[:, 10] - regression_target[:, 2])
        yt9 = (regression_target[:, 13] - regression_target[:, 5]) - (regression_target[:, 11] - regression_target[:, 3])
        xt10 = (regression_target[:, 12] - regression_target[:, 4]) - (regression_target[:, 14] - regression_target[:, 6])
        yt10 = (regression_target[:, 13] - regression_target[:, 5]) - (regression_target[:, 15] - regression_target[:, 7])
        xt11 = (regression_target[:, 12] - regression_target[:, 14]) - (regression_target[:, 4] - regression_target[:, 6])
        yt11 = (regression_target[:, 13] - regression_target[:, 15]) - (regression_target[:, 5] - regression_target[:, 7])
        xt12 = (regression_target[:, 12] - regression_target[:, 14]) - (regression_target[:, 10] - regression_target[:, 8])
        yt12 = (regression_target[:, 13] - regression_target[:, 15]) - (regression_target[:, 11] - regression_target[:, 9])
        orths_target = keras.backend.stack(
            [xt1, yt1, xt2, yt2, xt3, yt3, xt4, yt4, xt5, yt5, xt6, yt6, xt7, yt7, xt8, yt8, xt9, yt9, xt10, yt10, xt11, yt11, xt12, yt12],
            axis=1)

        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_xy = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )
        regression_orth = keras.losses.mean_absolute_error(orths, orths_target)

        #### compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        regression_loss_xy = keras.backend.sum(regression_xy) / normalizer
        regression_loss_orth = keras.backend.sum(regression_orth) / normalizer
        return weight * (weight_xy * regression_loss_xy + weight_orth * regression_loss_orth)

    return _orth_l1


def confidence_loss(num_classes=0, weight=1.0):

    def _confidence_loss(y_true, y_pred):

        # separate target and state

        regression_target = y_true[:, :, :, :-1]
        anchor_state      = y_true[:, :, :, -1]
        #regression_target, anchor_state = tf.split(y_true, num_or_size_splits=2, axis=3)
        #regression, confidence = tf.split(y_pred, num_or_size_splits=[num_classes * 7, 1], axis = 2)
        regression, confidence = tf.split(y_pred, num_or_size_splits=[-1, num_classes], axis=2)

        regression = tf.reshape(regression, tf.shape(regression_target))

        # filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        #regression        = backend.gather_nd(regression, indices)
        #confidence        = backend.gather_nd(confidence, indices)
        #regression_target = backend.gather_nd(regression_target, indices)

        #print('regression_target: ', regression_target[:, :, 16:])

        #tf.print('box: ', tf.math.reduce_mean(regression_target[:,:, :, 16]), tf.math.reduce_max(regression_target[:,:, :, 16]), tf.math.reduce_min(regression_target[:,:, :, 16]))
        #tf.print('pose: ', tf.math.reduce_mean(regression_target[:, :, :, 16:]),
        #         tf.math.reduce_max(regression_target[:, :, :, 16:]), tf.math.reduce_min(regression_target[:, :, :, 16:]))

        exp = tf.math.abs(regression - regression_target)
        exp = tf.math.reduce_sum(exp, axis=3)
        #exp_print = backend.gather_nd(exp, indices)
        #tf.print('exp: ', tf.math.reduce_mean(exp_print), tf.math.reduce_max(exp_print), tf.math.reduce_min(exp_print))
        #tf.print('conf: ', tf.math.reduce_max(confidence), tf.math.reduce_min(confidence))
        #conf_loss = 1.0 - tf.math.abs(tf.math.exp(-exp) - confidence)
        conf_loss = tf.math.abs(tf.math.exp(-1.0 * exp) - confidence)

        # comp norm per class
        normalizer = tf.math.reduce_sum(anchor_state, axis=[0, 1])
        conf_loss = tf.where(tf.math.equal(anchor_state, 1), conf_loss, 0.0)

        per_cls_loss = tf.math.reduce_sum(conf_loss, axis=[0, 1])
        loss = tf.math.divide_no_nan(per_cls_loss, normalizer)

        return weight * tf.math.reduce_sum(loss, axis=0)

    return _confidence_loss


def per_cls_l1_trans(num_classes=0, weight=1.0, sigma=3.0):

    sigma_squared = sigma ** 2

    def _per_cls_l1_trans(y_true, y_pred):
        regression_target = y_true[:, :, :, :-1]
        anchor_state = y_true[:, :, :, -1]

        in_shape = tf.shape(regression_target)
        anchor_state = tf.reshape(anchor_state, [in_shape[0] * in_shape[1], in_shape[2]])
        indices = tf.math.reduce_max(anchor_state, axis=1)
        indices = tf.where(tf.math.equal(indices, 1))[:, 0]

        y_pred_res = tf.reshape(y_pred, [in_shape[0] * in_shape[1], in_shape[2], in_shape[3]])
        regression = tf.gather(y_pred_res, indices, axis=0)
        y_true_res = tf.reshape(regression_target, [in_shape[0] * in_shape[1], in_shape[2], in_shape[3]])
        regression_target = tf.gather(y_true_res, indices, axis=0)

        regression = tf.transpose(regression, perm=[1, 2, 0])
        regression_target = tf.transpose(regression_target, perm=[1, 2, 0])

        regression_diff = regression_target - regression
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )
        per_cls_loss = tf.math.reduce_sum(regression_loss, axis=1)

        anchor_anno = tf.gather(anchor_state, indices, axis=0)
        anchor_anno = tf.transpose(anchor_anno, perm=[1, 0])

        regression_loss = tf.where(tf.math.equal(anchor_anno, 1), per_cls_loss, 0.0)
        per_cls_loss = tf.math.reduce_sum(regression_loss, axis=1)

        # comp norm per class
        normalizer = tf.math.reduce_sum(anchor_anno, axis=1) * tf.cast(num_classes, dtype=tf.float32) # accumulate over batch, locations and regressed values
        loss = tf.math.divide_no_nan(per_cls_loss, normalizer) # normalize per cls separately

        return weight * tf.math.reduce_sum(loss, axis=0)

    return _per_cls_l1_trans


def per_cls_l1(num_classes=0, weight=1.0, sigma=3.0):

    sigma_squared = sigma ** 2

    def _per_cls_l1(y_true, y_pred):
        regression_target = y_true[:, :, :, :-1]
        anchor_state = y_true[:, :, :, -1]

        in_shape = tf.shape(regression_target)
        anchor_state = tf.reshape(anchor_state, [in_shape[0] * in_shape[1], in_shape[2]])
        indices = tf.math.reduce_max(anchor_state, axis=1)
        indices = tf.where(tf.math.equal(indices, 1))[:, 0]

        y_pred_res = tf.reshape(y_pred, [in_shape[0] * in_shape[1], in_shape[3]])
        regression = tf.gather(y_pred_res, indices, axis=0)
        y_true_res = tf.reshape(regression_target, [in_shape[0] * in_shape[1], in_shape[2], in_shape[3]])
        regression_target = tf.gather(y_true_res, indices, axis=0)

        regression = tf.transpose(regression, perm=[1, 0])
        regression_target = tf.transpose(regression_target, perm=[1, 2, 0])

        regression_diff = regression_target - regression
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )
        per_cls_loss = tf.math.reduce_sum(regression_loss, axis=1)

        anchor_anno = tf.gather(anchor_state, indices, axis=0)
        anchor_anno = tf.transpose(anchor_anno, perm=[1, 0])

        regression_loss = tf.where(tf.math.equal(anchor_anno, 1), per_cls_loss, 0.0)
        per_cls_loss = tf.math.reduce_sum(regression_loss, axis=1)

        # comp norm per class
        normalizer = tf.math.reduce_sum(anchor_anno, axis=1) * tf.cast(in_shape[3], dtype=tf.float32) # accumulate over batch, locations and regressed values
        loss = tf.math.divide_no_nan(per_cls_loss, normalizer) # normalize per cls separately

        return weight * tf.math.reduce_sum(loss, axis=0)

    return _per_cls_l1


def per_cls_l1_sym(num_classes=0, weight=1.0, sigma=3.0):

    sigma_squared = sigma ** 2

    def _per_cls_l1_sym(y_true, y_pred, hyp_mask):


        regression_target = y_true[:, :, :, :, :-1]
        anchor_state = y_true[:, :, :, :, -1]

        in_shape = tf.shape(regression_target)
        anchor_state = tf.reshape(anchor_state, [in_shape[0] * in_shape[1], in_shape[2], in_shape[3]])
        indices = tf.math.reduce_max(anchor_state, axis=[1, 2])
        indices = tf.where(tf.math.equal(indices, 1))[:, 0]

        if hyp_mask == None:
            y_pred_res = tf.reshape(y_pred, [in_shape[0] * in_shape[1], in_shape[4]])
        else:
            y_pred_res = tf.reshape(y_pred, [in_shape[0] * in_shape[1], in_shape[2], in_shape[4]])

        regression = tf.gather(y_pred_res, indices, axis=0)
        y_true_res = tf.reshape(regression_target, [in_shape[0] * in_shape[1], in_shape[2], in_shape[3], in_shape[4]])
        regression_target = tf.gather(y_true_res, indices, axis=0)

        if hyp_mask == None:
            regression = tf.transpose(regression, perm=[1, 0])
        else:
            regression = tf.transpose(regression, perm=[1, 2, 0])
        regression_target = tf.transpose(regression_target, perm=[2, 1, 3, 0])

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression_target - regression
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )
        # symmetry-handling by re-weighting
        # loss:  [8 33 3 1067]
        # anchor:  [25200 33 8]
        # indices:  [1067]
        # don't want sparse tensors
        # L = 2 * l ((min+max) - x) * (1/(min+max))
        regression_loss = tf.math.reduce_sum(regression_loss, axis=2)
        if hyp_mask == None:
            minmax = (tf.math.reduce_min(regression_loss, axis=0) + tf.math.reduce_max(regression_loss, axis=0))
            scaler = (tf.math.divide_no_nan(1.0, minmax)) * (minmax - regression_loss)
            regression_loss = regression_loss * scaler * 2.0

        regression_loss = tf.transpose(regression_loss, perm=[2, 1, 0])
        anchor_anno = tf.gather(anchor_state, indices, axis=0)
        pose_hyp_anno = tf.math.reduce_sum(anchor_anno, axis=2)
        regression_loss = tf.where(tf.math.equal(anchor_anno, 1.0), regression_loss, 0.0)

        if hyp_mask == None:
            regression_loss = tf.math.reduce_sum(regression_loss, axis=2)
            scaler = tf.transpose(scaler, perm=[2, 1, 0])
            scaler_mask = tf.tile(tf.math.reduce_max(scaler, axis=2)[:, :, tf.newaxis], [1, 1, 8])
            hyp_mask = tf.where(tf.math.equal(scaler, scaler_mask), 1.0, 0.0)
            hyp_mask = tf.where(tf.math.equal(anchor_anno, 0.0), 0.0, hyp_mask)
            #tf.print('scaler: ', tf.math.reduce_min(scaler), tf.math.reduce_max(scaler))
        else:
            regression_loss = tf.where(tf.math.equal(hyp_mask, 1.0), regression_loss, 0.0)
            regression_loss = tf.math.reduce_sum(regression_loss, axis=2)
            pose_hyp_anno = tf.math.reduce_sum(hyp_mask, axis=2)

        regression_loss = tf.math.divide_no_nan(regression_loss, pose_hyp_anno)

        per_cls_loss = tf.math.reduce_sum(regression_loss, axis=0)
        normalizer = tf.math.reduce_max(anchor_state, axis=2)
        normalizer = tf.math.reduce_sum(normalizer, axis=0) * tf.cast(num_classes, dtype=tf.float32)
        # * tf.cast(num_classes, dtype=tf.float32)
        #* tf.cast(in_shape[4], dtype=tf.float32)  # accumulate over batch, locations and regressed values
        loss = tf.math.divide_no_nan(per_cls_loss, normalizer)

        #tf.print('per_cls_loss: ', per_cls_loss)
        #tf.print('normalizer: ', normalizer)

        #vanilla best sym_hyp
        #regression_loss = tf.math.reduce_min(regression_loss, axis=0) # reduce regression loss to min hypothesis
        #per_cls_loss = tf.math.reduce_sum(regression_loss, axis=1)

        #anchor_anno = tf.math.reduce_max(anchor_state, axis=2)
        #anchor_anno = tf.gather(anchor_anno, indices, axis=0)
        #anchor_anno = tf.transpose(anchor_anno, perm=[1, 0])

        #regression_loss = tf.where(tf.math.equal(anchor_anno, 1), per_cls_loss, 0.0)
        #per_cls_loss = tf.math.reduce_sum(regression_loss, axis=1)

        # comp norm per class
        #normalizer = tf.math.reduce_max(anchor_state, axis=2) # reduce normalizer to single hypothesis per location
        #normalizer = tf.math.reduce_sum(normalizer, axis=0) * tf.cast(in_shape[4], dtype=tf.float32) # accumulate over batch, locations and regressed values
        #loss = tf.math.divide_no_nan(per_cls_loss, normalizer) # normalize per cls separately

        return weight * tf.math.reduce_sum(loss, axis=0), hyp_mask

    return _per_cls_l1_sym


def per_cls_l1_rep(num_classes=0, weight=1.0, sigma=3.0):

    sigma_squared = sigma ** 2

    def _per_cls_l1_rep(y_true, y_pred, hyp_mask):

        regression_target = y_true[:, :, :, :, :-1]
        anchor_state = y_true[:, :, :, :, -1]

        in_shape = tf.shape(regression_target)
        anchor_state = tf.reshape(anchor_state, [in_shape[0] * in_shape[1], in_shape[2], in_shape[3]])
        indices = tf.math.reduce_max(anchor_state, axis=[1, 2])
        indices = tf.where(tf.math.equal(indices, 1))[:, 0]

        y_pred_res = tf.reshape(y_pred, [in_shape[0] * in_shape[1], in_shape[2], in_shape[4]])
        regression = tf.gather(y_pred_res, indices, axis=0)
        y_true_res = tf.reshape(regression_target, [in_shape[0] * in_shape[1], in_shape[2], in_shape[3], in_shape[4]])
        regression_target = tf.gather(y_true_res, indices, axis=0)

        regression = tf.transpose(regression, perm=[1, 2, 0])
        regression_target = tf.transpose(regression_target, perm=[2, 1, 3, 0])

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression_target - regression
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )
        # symmetry-handling by re-weighting
        # loss:  [8 33 3 1067]
        # anchor:  [25200 33 8]
        # indices:  [1067]
        # don't want sparse tensors
        # L = 2 * l ((min+max) - x) * (1/(min+max))
        regression_loss = tf.math.reduce_sum(regression_loss, axis=2)

        regression_loss = tf.transpose(regression_loss, perm=[2, 1, 0])
        anchor_anno = tf.gather(anchor_state, indices, axis=0)
        regression_loss = tf.where(tf.math.equal(anchor_anno, 1.0), regression_loss, 0.0)

        regression_loss = tf.where(tf.math.equal(hyp_mask, 1.0), regression_loss, 0.0)
        regression_loss = tf.math.reduce_sum(regression_loss, axis=2)
        pose_hyp_anno = tf.math.reduce_sum(hyp_mask, axis=2)

        regression_loss = tf.math.divide_no_nan(regression_loss, pose_hyp_anno)

        per_cls_loss = tf.math.reduce_sum(regression_loss, axis=0)
        normalizer = tf.math.reduce_max(anchor_state, axis=2)
        normalizer = tf.math.reduce_sum(normalizer, axis=0) * tf.cast(num_classes, dtype=tf.float32)

        loss = tf.math.divide_no_nan(per_cls_loss, normalizer)

        return weight * tf.math.reduce_sum(loss, axis=0)

    return _per_cls_l1_rep


def projection_deviation(num_classes=0, weight=1.0, sigma=3.0):

    sigma_squared = sigma ** 2

    def _projection_deviation(y_true, y_pred):

        in_shape_gt = tf.shape(y_true)
        in_shape_es = tf.shape(y_pred)
        anchor_state = tf.reshape(y_true, [in_shape_gt[0] * in_shape_gt[1], in_shape_gt[2]])
        indices = tf.math.reduce_max(anchor_state, axis=1)
        indices = tf.where(tf.math.equal(indices, 1))[:, 0]

        regression = tf.reshape(y_pred, [in_shape_es[0] * in_shape_es[1], in_shape_es[2], in_shape_es[3]])
        regression = tf.gather(regression, indices, axis=0)

        #regression_diff = keras.backend.abs(regression) #* 0.01
        regression_diff = regression
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )
        per_loc_loss = tf.math.reduce_sum(regression_loss, axis=2) # accumulate over points

        anchor_anno = tf.gather(anchor_state, indices, axis=0)
        regression_loss = tf.where(tf.math.equal(anchor_anno, 1), per_loc_loss, 0.0)

        # comp norm per class
        normalizer = tf.math.reduce_sum(anchor_state, axis=0)  * tf.cast(num_classes, dtype=tf.float32)
        per_cls_loss = tf.math.reduce_sum(regression_loss, axis=[0])

        loss = tf.math.divide_no_nan(per_cls_loss, normalizer)

        return weight * tf.math.reduce_sum(loss, axis=0)

    return _projection_deviation

'''
def projection_deviation(num_classes=0, weight=1.0, sigma=3.0):

    sigma_squared = sigma ** 2

    def _projection_deviation(y_true, y_pred):

        boxes, reprojection = tf.split(y_pred, num_or_size_splits=2, axis=3)
        regression = (boxes-reprojection) * 0.01 # requires some scaling due to magnitude of pixel space location

        anchor_state = y_true[:, :, :, 16:]
        regression_target = y_true[:, :, :, :16]
        # tf.where faster than element-wise multiplication
        regression = tf.where(tf.math.equal(anchor_state, 1), regression[:, :, :, :16], 0.0)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # comp norm per class
        normalizer = tf.math.reduce_sum(anchor_state, axis=[0, 1, 3])
        #retain per cls loss
        per_cls_loss = tf.math.reduce_sum(regression_loss, axis=[0, 1, 3])

        loss = tf.math.divide_no_nan(per_cls_loss, normalizer)

        return weight * tf.math.reduce_sum(loss, axis=0)

    return _projection_deviation
'''
