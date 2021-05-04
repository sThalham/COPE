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

import keras
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


def focal_mask(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha. vanilla 0.25 2.0
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal_mask(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
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

        cls_loss = 0.1 * focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal_mask


def cross(weight=50.0):

    def _cross(y_true, y_pred):
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)

        #cls_loss = weight * keras.losses.binary_crossentropy(labels, classification)
        cls_loss = weight * keras.losses.categorical_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])  # usually for regression
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        #normalizer = backend.where(keras.backend.equal(anchor_state, 1))       # usually for classification
        #normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        #normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _cross


def smooth_l1(sigma=3.0):
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
        return loss

    return _smooth_l1


def smooth_l1_pose(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1_pose(y_true, y_pred):
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
        return 0.1 * loss

    return _smooth_l1_pose


def weighted_mse(weight=60.0):

    def _wMSE(y_true, y_pred):

        regression        = y_pred
        regression_target = y_true[:, :, :, :-1]
        anchor_state      = y_true[:, :, :, -1]

        # somethings fucky here
        #### filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        regression_loss = weight * keras.losses.mean_squared_error(regression, regression_target)

        #### compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return regression_loss / normalizer

    return _wMSE


def weighted_l1(weight=1.6):

    def _wl1(y_true, y_pred):

        regression        = y_pred
        regression_target = y_true[:, :, :, :-1]
        anchor_state      = y_true[:, :, :, -1]

        # somethings fucky here
        #### filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        regression_loss = weight * keras.losses.mean_absolute_error(regression, regression_target)

        #### compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _wl1


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


def orthogonal_l1(weight=0.125, sigma=3.0):

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


def smooth_l1_xy(sigma=3.0, weight=0.1):
    sigma_squared = sigma ** 2

    def _smooth_l1_xy(y_true, y_pred):
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :, :-1]
        anchor_state      = y_true[:, :, :, -1]

        # filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = weight * backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1_xy

