import tensorflow.keras as keras
import tensorflow as tf

from . import model
from . import Backbone
from ..utils.image import preprocess_image


def replace_relu_with_swish(model):
    for layer in tuple(model.layers[1:]):
        layer_type = type(layer).__name__
        if hasattr(layer, 'activation') and layer.activation.__name__ == 'relu':
            if layer_type == "Conv2D":
                # conv layer with swish activation
                layer.activation = tf.keras.activations.swish
            else:
                # activation layer
                layer.activation = tf.keras.activations.swish
    return model


class XceptionBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(XceptionBackbone, self).__init__(backbone)
        self.custom_objects.update()

    def model(self, *args, **kwargs):
        return xception_model(*args, **kwargs)

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def xception_model(num_classes, obj_diameters, correspondences=None, intrinsics=None, inputs=None, modifier=None, **kwargs):
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            # inputs = keras.layers.Input(shape=(None, None, 3))
            inputs = keras.layers.Input(shape=(480, 640, 3))

    xception = tf.keras.applications.Xception(
        include_top=False, weights='imagenet', input_tensor=inputs, classes=num_classes)

    #net = tfcv_get_model("seresnext50_32x4d", pretrained=True, data_format="channels_last")
    #seresnext50 = net(inputs)

    for i, layer in enumerate(xception.layers):
        # if i < 39 and 'bn' not in layer.name: #freezing first 2 stages
        #    layer.trainable=False
        if i < 22 or 'bn' in layer.name:  # freezing first 2 stages
            layer.trainable = False
            #print(i, layer.name, layer.type)

    #xception = replace_relu_with_swish(xception)

        # invoke modifier if given
    if modifier:
        xception = modifier(xception)

    xception_outputs = [xception.layers[29].output, xception.layers[119].output, xception.layers[129].output]

    # create the full model
    return model.cope(inputs=inputs, num_classes=num_classes, obj_correspondences=correspondences, obj_diameters=obj_diameters, intrinsics=intrinsics, backbone_layers=xception_outputs, **kwargs)


