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
from tf2cv.model_provider import get_model as tf2cv_get_model

from . import model
from . import Backbone
from ..utils.image import preprocess_image


def replace_relu_with_swish(model):
    for layer in tuple(model.layers):
        layer_type = type(layer).__name__
        if hasattr(layer, 'activation') and layer.activation.__name__ == 'relu':
            if layer_type == "Conv2D":
                # conv layer with swish activation
                layer.activation = tf.keras.activations.swish
            else:
                # activation layer
                layer.activation = tf.keras.activations.swish
    return model


# taken from https://github.com/broadinstitute/keras-resnet/blob/master/keras_resnet/layers/_batch_normalization.py
class BatchNormalization_freezeable(keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
    """
    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization_freezeable, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, *args, **kwargs):
        # Force test mode if frozen, otherwise use default behaviour (i.e., training=None).
        if self.freeze:
            kwargs['training'] = False
        return super(BatchNormalization_freezeable, self).call(*args, **kwargs)

    def get_config(self):
        config = super(BatchNormalization_freezeable, self).get_config()
        config.update({'freeze': self.freeze})
        return config


class DarkNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(DarkNetBackbone, self).__init__(backbone)
        self.custom_objects.update()

    def model(self, *args, **kwargs):
        """ Returns PyraPose using the correct backbone.
        """
        return darknet_model(*args, **kwargs)

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def darknet_model(num_classes, inputs=None, modifier=None, **kwargs):
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            # inputs = keras.layers.Input(shape=(None, None, 3))
            inputs = keras.layers.Input(shape=(480, 640, 3))

    darknet = tf2cv_get_model("darknet53", pretrained=True, data_format="channels_last")

    for i, layer in enumerate(darknet.layers):
        # if i < 39 and 'bn' not in layer.name: #freezing first 2 stages
        #    layer.trainable=False
        if i < 39 or 'bn' in layer.name:  # freezing first 2 stages
            layer.trainable = False
        print(i, layer.name)

        # if 'bn' in layer.name:
        #    layer.trainable = False
        #    print("weights:", len(layer.weights))
        #    print("trainable_weights:", len(layer.trainable_weights))
        #    print("non_trainable_weights:", len(layer.non_trainable_weights))

    #resnet = replace_relu_with_swish(resnet)

        # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    resnet_outputs = [resnet.layers[80].output, resnet.layers[142].output, resnet.layers[174].output]

    # create the full model
    return model.pyrapose(inputs=inputs, num_classes=num_classes, backbone_layers=resnet_outputs, **kwargs)


