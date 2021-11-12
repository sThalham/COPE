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
from ..backend import resize_images, transpose, shift, bbox_transform_inv, clip_by_value, box3D_transform_inv, box3D_denorm, poses_denorm
from ..utils import anchors as utils_anchors

import numpy as np
from tensorflow import meshgrid


class Locations(keras.layers.Layer):
    """ Keras layer for generating locations for a given shape.
    """

    def __init__(self, stride=[8, 16, 32], *args, **kwargs):
        """ Initializer for an Locations layer.
        """
        self.stride = stride
        super(Locations, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = keras.backend.shape(features)

        # generate proposals from bbox deltas and shifted anchors
        if keras.backend.image_data_format() == 'channels_first':
            shape = features_shape[2:4]
            #anchors = shift(features_shape[2:4], self.stride, self.anchors)
        else:
            shape = features_shape[1:3]
            #anchors = shift(features_shape[1:3], self.stride, self.anchors)

        shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * self.stride
        shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * self.stride

        shift_x, shift_y = meshgrid(shift_x, shift_y)
        shift_x = keras.backend.reshape(shift_x, [-1])
        shift_y = keras.backend.reshape(shift_y, [-1])

        shifts = keras.backend.stack([
            shift_x,
            shift_y
        ], axis=0)

        shifts = keras.backend.transpose(shifts)
        k = keras.backend.shape(shifts)[0]

        #shifts = keras.backend.reshape(shifts, [1, k, 2]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 2]), keras.backend.floatx())
        shifts = keras.backend.cast(keras.backend.reshape(shifts, [1, k, 2]), keras.backend.floatx())
        #shifted_anchors = keras.backend.reshape(shifts, [k, 2])
        #anchors = keras.backend.tile(keras.backend.expand_dims(shifted_anchors, axis=0), (features_shape[0], 1, 1))
        return shifts

        #anchors = keras.backend.tile(shifts, (features_shape[0], 1, 1))
        #return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            if keras.backend.image_data_format() == 'channels_first':
                total = np.prod(input_shape[2:4])
            else:
                total = np.prod(input_shape[1:3])

            return (input_shape[0], total, 2)
        else:
            return (input_shape[0], None, 2)

    def get_config(self):
        config = super(Locations, self).get_config()
        #config.update({})

        return config


class Locations_Hacked(keras.layers.Layer):
    """ Keras layer for generating locations for a given shape.
    """

    def __init__(self, shape=[[60, 80], [30, 40], [15, 20]], stride=[8, 16, 32], *args, **kwargs):
        """ Initializer for an Locations layer.
        """
        self.stride = stride
        self.shape = shape
        super(Locations_Hacked, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = keras.backend.shape(features)

        shift_x = (keras.backend.arange(0, self.shape[0][1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5,
                                                                                                      dtype=keras.backend.floatx())) * self.stride[0]
        shift_y = (keras.backend.arange(0, self.shape[0][0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5,
                                                                                                      dtype=keras.backend.floatx())) * self.stride[0]
        shift_x, shift_y = meshgrid(shift_x, shift_y)
        shift_x = keras.backend.reshape(shift_x, [-1])
        shift_y = keras.backend.reshape(shift_y, [-1])
        shifts = keras.backend.stack([
            shift_x,
            shift_y
        ], axis=0)
        shifts = keras.backend.transpose(shifts)
        k = keras.backend.shape(shifts)[0]
        shifts_P3 = keras.backend.cast(keras.backend.reshape(shifts, [1, k, 2]), keras.backend.floatx())

        shift_x = (keras.backend.arange(0, self.shape[1][1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5,
                                                                                                      dtype=keras.backend.floatx())) * self.stride[1]
        shift_y = (keras.backend.arange(0, self.shape[1][0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5,
                                                                                                      dtype=keras.backend.floatx())) * self.stride[1]
        shift_x, shift_y = meshgrid(shift_x, shift_y)
        shift_x = keras.backend.reshape(shift_x, [-1])
        shift_y = keras.backend.reshape(shift_y, [-1])
        shifts = keras.backend.stack([
            shift_x,
            shift_y
        ], axis=0)
        shifts = keras.backend.transpose(shifts)
        k = keras.backend.shape(shifts)[0]
        shifts_P4 = keras.backend.cast(keras.backend.reshape(shifts, [1, k, 2]), keras.backend.floatx())

        shift_x = (keras.backend.arange(0, self.shape[2][1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5,
                                                                                                      dtype=keras.backend.floatx())) * self.stride[2]
        shift_y = (keras.backend.arange(0, self.shape[2][0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5,
                                                                                                      dtype=keras.backend.floatx())) * self.stride[2]
        shift_x, shift_y = meshgrid(shift_x, shift_y)
        shift_x = keras.backend.reshape(shift_x, [-1])
        shift_y = keras.backend.reshape(shift_y, [-1])
        shifts = keras.backend.stack([
            shift_x,
            shift_y
        ], axis=0)
        shifts = keras.backend.transpose(shifts)
        k = keras.backend.shape(shifts)[0]
        shifts_P5 = keras.backend.cast(keras.backend.reshape(shifts, [1, k, 2]), keras.backend.floatx())

        shifts = keras.layers.Concatenate(axis=1)([shifts_P3, shifts_P4, shifts_P5])
        anchors = keras.backend.tile(shifts, (features_shape[0], 1, 1))
        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            return (input_shape[0], 6300, 2)
        else:
            return (input_shape[0], None, 2)

    def get_config(self):
        config = super(Locations_Hacked, self).get_config()
        #config.update({})

        return config


class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == 'channels_first':
            source = transpose(source, (0, 2, 3, 1))
            output = resize_images(source, (target_shape[2], target_shape[3]), method='nearest')
            output = transpose(output, (0, 3, 1, 2))
            return output
        else:
            return resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class RegressBoxes(keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.
    """

    def __init__(self, mean=None, std=None, *args, **kwargs):
        """ Initializer for the RegressBoxes layer.

        Args
            mean: The mean value of the regression values which was used for normalization.
            std: The standard value of the regression values which was used for normalization.
        """
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.full(16, 0.65)

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        self.mean = mean
        self.std  = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std' : self.std.tolist(),
        })

        return config


class RegressBoxes3D(keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.
    """

    def __init__(self, mean=None, std=None, *args, **kwargs):
        """ Initializer for the RegressBoxes layer.

        Args
            mean: The mean value of the regression values which was used for normalization.
            std: The standard value of the regression values which was used for normalization.
        """
        if mean is None:
            mean = np.full(16, 0)  # np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if std is None:
            std = np.full(16, 0.65)
        else:
            std = np.full(16, std)
        if mean is None:
            raise ValueError('Object diameters are required for de-standardization.')

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        self.mean = mean
        self.std  = std
        super(RegressBoxes3D, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        regression, locations, diameters = inputs
        return box3D_transform_inv(regression, locations, diameters, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = super(RegressBoxes3D, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std' : self.std.tolist(),
        })

        return config


class DenormRegression(keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.
    """

    def __init__(self, mean=None, std=None, *args, **kwargs):
        """ Initializer for the RegressBoxes layer.

        Args
            mean: The mean value of the regression values which was used for normalization.
            std: The standard value of the regression values which was used for normalization.
        """
        if mean is None:
            mean = np.full(16, 0)  # np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if std is None:
            std = np.full(16, 0.65)
        else:
            std = np.full(16, std)
        if mean is None:
            raise ValueError('Object diameters are required for de-standardization.')

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        self.mean = mean
        self.std  = std
        #self.obj_diameters = diameter_tensor
        super(DenormRegression, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        regression, locations = inputs
        return box3D_denorm(regression, locations, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = super(DenormRegression, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std' : self.std.tolist(),
        })

        return config


class DenormPoses(keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.
    """

    def __init__(self, *args, **kwargs):

        super(DenormPoses, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        pose_regression = inputs
        return poses_denorm(pose_regression)

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = super(DenormPoses, self).get_config()

        return config


class ClipBoxes(keras.layers.Layer):
    """ Keras layer to clip box values to lie inside a given shape.
    """

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())
        if keras.backend.image_data_format() == 'channels_first':
            height = shape[2]
            width  = shape[3]
        else:
            height = shape[1]
            width  = shape[2]
        x1 = clip_by_value(boxes[:, :, 0], 0, width)
        y1 = clip_by_value(boxes[:, :, 1], 0, height)
        x2 = clip_by_value(boxes[:, :, 2], 0, width)
        y2 = clip_by_value(boxes[:, :, 3], 0, height)

        return keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]



