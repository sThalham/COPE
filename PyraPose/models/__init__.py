from __future__ import print_function
import sys
import tensorflow


class Backbone(object):
    """ This class stores additional information on backbones.
    """
    def __init__(self, backbone):
        # a dictionary mapping custom layer names to the correct classes
        from .. import layers
        from .. import losses
        from .. import initializers
        from . import model
        self.custom_objects = {
            'UpsampleLike'     : layers.UpsampleLike,
            'PriorProbability' : initializers.PriorProbability,
            'RegressBoxes'     : layers.RegressBoxes,
            'FilterDetections' : layers.FilterDetections,
            'ClipBoxes'        : layers.ClipBoxes,
            '_smooth_l1'       : losses.smooth_l1(),
            '_smooth_l1_weighted'  : losses.smooth_l1_weighted(),
            '_focal'           : losses.focal(),
            '_focal_mask'      : losses.focal_mask(),
            '_cross'           : losses.cross(),
            '_wMSE'            : losses.weighted_mse(),
            '_wl1'            : losses.weighted_l1(),
            '_msle'           : losses.weighted_msle(),
            '_orth_l1'         : losses.orthogonal_l1(),
            'RegressBoxes3D'   : layers.RegressBoxes3D(),
            '_focal_l1'            : losses.focal_l1(),
        }

        self.backbone = backbone

    def model(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        raise NotImplementedError('retinanet method not implemented.')

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        raise NotImplementedError('preprocess_image method not implemented.')


def backbone(backbone_name):
    if 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone))

    return b(backbone_name)


def load_model(filepath, backbone_name='resnet'):
    import tensorflow.keras.models

    print(backbone(backbone_name).custom_objects)

    return tensorflow.keras.models.load_model(filepath, custom_objects=backbone(backbone_name).custom_objects)


def convert_model(model):
    from .model import inference_model
    return inference_model(model=model)


def assert_training_model(model):
    assert (all(output in model.output_names for output in ['points', 'cls', 'center'])), "Input is not a training model. Outputs were found, outputs are: {}).".format(model.output_names)


def check_training_model(model):
    try:
        assert_training_model(model)
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
