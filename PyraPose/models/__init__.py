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
            'UpsampleLike'              : layers.UpsampleLike,
            'PriorProbability'          : initializers.PriorProbability,
            'RegressBoxes'              : layers.RegressBoxes,
            'FilterDetections'          : layers.FilterDetections,
            'ClipBoxes'                 : layers.ClipBoxes,
            '_smooth_l1'                : losses.smooth_l1(),
            '_focal'                    : losses.focal(),
            '_orth_l1'                  : losses.orthogonal_l1(),
            'RegressBoxes3D'            : layers.RegressBoxes3D(),
            'ProjectBoxes'              :layers.ProjectBoxes(),
            'DenormRegression'          : layers.DenormRegression(),
            'NormRegression'          : layers.NormRegression(),
            'Locations'                 : layers.Locations(),
            'Locations_Hacked'          : layers.Locations_Hacked(),
            '_per_cls_l1'               : losses.per_cls_l1(),
            '_per_cls_l1_trans'               : losses.per_cls_l1_trans(),
            '_per_cls_l1_pose'         : losses.per_cls_l1_trans(),
            '_per_cls_l1_sym'               : losses.per_cls_l1_sym(),
            '_per_cls_l1_rep'           : losses.per_cls_l1_rep(),
            '_projection_deviation'     : losses.projection_deviation()
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
    elif 'resnet152' in backbone_name:
        from .resnet152 import ResNetBackbone as b
    elif 'efficientnet' in backbone_name:
        from .efficientnet import EfficientNetBackbone as b
    elif 'darknet' in backbone_name:
        from .darknet53 import DarkNetBackbone as b
    elif 'xception' in backbone_name:
        from .xception import XceptionBackbone as b
    elif 'nasnetmobile' in backbone_name:
        from .nasnetmobile import NASNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone))

    return b(backbone_name)


def load_model(filepath, backbone_name='resnet50'):
    import tensorflow.keras.models

    return tensorflow.keras.models.load_model(filepath, custom_objects=backbone(backbone_name).custom_objects)


def convert_model(model, diameters, classes):
    from .model import inference_model
    return inference_model(model=model, object_diameters=diameters, num_classes=classes)


def assert_training_model(model):
    assert (all(output in model.output_names for output in ['pts', 'box', 'cls', 'tra', 'rot', 'pro', 'con'])), "Input is not a training model. Outputs were found, outputs are: {}).".format(model.output_names)


def check_training_model(model):
    try:
        assert_training_model(model)
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
