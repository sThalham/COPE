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
            '_residual_loss'         : losses.residual_loss(),
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
    if 'resnet50' in backbone_name:
        from .resnet50 import ResNetBackbone as b
    elif 'resnet101' in backbone_name:
        from .resnet101 import ResNetBackbone as b
    elif 'efficientnet' in backbone_name:
        from .efficientnet import EfficientNetBackbone as b
    elif 'darknet' in backbone_name:
        from .darknet53 import DarkNetBackbone as b
    elif 'xception' in backbone_name:
        from .xception import XceptionBackbone as b
    elif 'densenet' in backbone_name:
        from .densenet import DenseNetBackbone as b
    elif 'nasnetmobile' in backbone_name:
        from .nasnetmobile import NASNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone))

    return b(backbone_name)


def load_model(filepath, backbone_name='resnet50'):
    import tensorflow.keras.models

    backbone_name='resnet101'

    return tensorflow.keras.models.load_model(filepath, custom_objects=backbone(backbone_name).custom_objects)


def convert_model(model):
    from .model import inference_model
    return inference_model(model=model)


def assert_training_model(model):
    #assert (all(output in model.output_names for output in ['points', 'conf', 'cls', 'center'])), "Input is not a training model. Outputs were found, outputs are: {}).".format(model.output_names)
    assert (all(output in model.output_names for output in ['points', 'res', 'cls'])), "Input is not a training model. Outputs were found, outputs are: {}).".format(model.output_names)
    #assert (all(output in model.output_names for output in ['points', 'cls', 'P3', 'P4', 'P5'])), "Input is not a training model. Outputs were found, outputs are: {}).".format(model.output_names)


def check_training_model(model):
    try:
        assert_training_model(model)
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
