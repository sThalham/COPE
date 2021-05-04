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
import keras_efficientnets

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image
from keras_efficientnets import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4


class EffNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(EffNetBackbone, self).__init__(backbone)
        #self.custom_objects.update(keras_efficientnets.cus)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return effnet_retinanet(*args, **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['efficientnetsB0', 'efficientnetsB1', 'efficientnetsB2', 'efficientnetsB3', 'efficientnetsB4']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def effnet_retinanet(num_classes, inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a resnet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('resnet50', 'resnet101', 'resnet152')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a ResNet backbone.
    """
    # choose default input
    #num_classes = 1
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            #inputs = keras.layers.Input(shape=(3, None, None))
            inputs_0 = keras.layers.Input(shape=(3, None, None))
            inputs_1 = keras.layers.Input(shape=(3, None, None))
        else:
            #inputs = keras.layers.Input(shape=(None, None, 3))
            inputs_0 = keras.layers.Input(shape=(480, 640, 3))
            inputs_1 = keras.layers.Input(shape=(480, 640, 3))

    effnet_rgb = keras_efficientnets.EfficientNetB1(include_top=False, weights='imagenet', input_tensor=inputs_0,
                                                    pooling=None, classes=num_classes)
    effnet_dep = keras_efficientnets.EfficientNetB1(include_top=False, weights='imagenet', input_tensor=inputs_1,
                                                pooling=None, classes=num_classes)

    print(effnet_dep.summary())

    # invoke modifier if given
    if modifier:
        effnet_rgb = modifier(effnet_rgb)
        effnet_dep = modifier(effnet_dep)

    for i, layer in enumerate(effnet_dep.layers):
        print(i, layer.name, layer)
    #    layer.trainable = False

    layer_names = [117, 235, 338]  # EfficientnetsB1, B2
    #layer_names = [117, 265, 383] # EffNet B3 swish_102, _132, _156
    #layer_names = [147, 325, 473] # B4
    layer_outputs_rgb = [effnet_rgb.layers[idx].output for idx in layer_names]
    layer_outputs_dep = [effnet_dep.layers[idx].output for idx in layer_names]
    print(layer_outputs_rgb)

    return retinanet.retinanet(inputs=[inputs_0, inputs_1], num_classes=num_classes,
                               backbone_layers_rgb=layer_outputs_rgb, backbone_layers_dep=layer_outputs_dep, **kwargs)

