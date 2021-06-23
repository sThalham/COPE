import tensorflow.keras as keras
import tensorflow as tf
from .. import initializers
from .. import layers
from . import assert_training_model


def default_classification_model(
    num_classes,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
):
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))

    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    labels = keras.layers.Conv2D(
        filters=num_classes,
        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        labels = keras.layers.Permute((2, 3, 1))(labels)
    labels = keras.layers.Reshape((-1, num_classes))(labels)
    labels = keras.layers.Activation('sigmoid')(labels)

    # centerness = keras.layers.Conv2D(
    #     filters=1,
    #     kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
    #     bias_initializer=initializers.PriorProbability(probability=prior_probability),
    #     **options
    # )(outputs)
    #
    # # reshape output and apply sigmoid
    # if keras.backend.image_data_format() == 'channels_first':
    #     centerness = keras.layers.Permute((2, 3, 1))(centerness)
    # centerness = keras.layers.Reshape((-1, 1))(centerness) # centerness = keras.layers.Reshape((-1, num_classes))(centerness)
    # centerness = keras.layers.Activation('sigmoid')(centerness)

    # return keras.models.Model(inputs=inputs, outputs=labels), keras.models.Model(inputs=inputs, outputs=centerness)
    return keras.models.Model(inputs=inputs, outputs=labels)


def default_regression_model(num_values, pyramid_feature_size=256, prior_probability=0.01, regression_feature_size=512):
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros',
        'kernel_regularizer' : keras.regularizers.l2(0.001),
    }

    options_centerness = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))

    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            **options
        )(outputs)
        tf.print('layer: ', outputs.name)

    regress = keras.layers.Conv2D(num_values, **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        regress = keras.layers.Permute((2, 3, 1))(regress)
    regress = keras.layers.Reshape((-1, num_values))(regress)

    centerness = keras.layers.Conv2D(
        filters=1,
        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        **options_centerness
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        centerness = keras.layers.Permute((2, 3, 1))(centerness)
    centerness = keras.layers.Reshape((-1, 1))(centerness) # centerness = keras.layers.Reshape((-1, num_classes))(centerness)
    centerness = keras.layers.Activation('sigmoid')(centerness)

    #return keras.models.Model(inputs=inputs, outputs=regress)
    return keras.models.Model(inputs=inputs, outputs=regress), keras.models.Model(inputs=inputs, outputs=centerness)


def __create_PFPN(C3, C4, C5, feature_size=256):
    
    # 3x3 conv for test 4
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3)
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4)
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5)

    P5_upsampled = layers.UpsampleLike()([P5, P4])
    P4_upsampled = layers.UpsampleLike()([P4, P3])
    P4_mid = keras.layers.Add()([P5_upsampled, P4])
    P4_mid = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P4_mid)    # replace with depthwise and 3x1+1x3
    P3_mid = keras.layers.Add()([P4_upsampled, P3])
    P3_mid = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P3_mid)    # replace with depthwise and 3x1+1x3
    P3_down = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(P3_mid)
    P3_fin = keras.layers.Add()([P3_mid, P3])  # skip connection
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3_fin) # replace with depthwise and 3x1+1x3

    P4_fin = keras.layers.Add()([P3_down, P4_mid])
    P4_down = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(P4_mid)
    P4_fin = keras.layers.Add()([P4_fin, P4])  # skip connection
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4_fin) # replace with depthwise and 3x1+1x3

    P5_fin = keras.layers.Add()([P4_down, P5])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5_fin) # replace with depthwise and 3x1+1x3

    return [P3, P4, P5]


def pyrapose(
    inputs,
    backbone_layers,
    num_classes,
    create_pyramid_features = __create_PFPN,
    name                    = 'pyrapose'
):
    regression_branch = default_regression_model(16)
    location_branch = default_classification_model(num_classes)

    b1, b2, b3 = backbone_layers

    '''
    features = create_pyramid_features(b1, b2, b3)
    pyramids = []
    regression_P3 = regression_branch[0](features[0])
    regression_P4 = regression_branch[0](features[1])
    regression_P5 = regression_branch[0](features[2])

    location_P3 = location_branch(features[0])
    location_P4 = location_branch(features[1])
    location_P5 = location_branch(features[2])

    center_P3 = regression_branch[1](features[0])
    center_P4 = regression_branch[1](features[1])
    center_P5 = regression_branch[1](features[2])
    '''

    P3, P4, P5 = create_pyramid_features(b1, b2, b3)
    pyramids = []
    regression_P3 = regression_branch[0](P3)
    regression_P4 = regression_branch[0](P4)
    regression_P5 = regression_branch[0](P5)

    location_P3 = location_branch(P3)
    location_P4 = location_branch(P4)
    location_P5 = location_branch(P5)

    center_P3 = regression_branch[1](P3)
    center_P4 = regression_branch[1](P4)
    center_P5 = regression_branch[1](P5)

    pyramids.append(keras.layers.Concatenate(axis=1, name='points')([regression_P3, regression_P4, regression_P5]))
    pyramids.append(keras.layers.Concatenate(axis=1, name='cls')([location_P3, location_P4, location_P5]))
    pyramids.append(keras.layers.Concatenate(axis=1, name='center')([center_P3, center_P4, center_P5]))

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def __build_locations(features):
    strides = [8, 16, 32]
    locations = [
        layers.Locations(stride=strides[i], name='locations_{}'.format(i))(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='locations')(locations)


def inference_model(
    model                 = None,
    name                  = 'pyrapose',
    **kwargs
):

    # create RetinaNet model
    if model is None:
        model = retinanet(**kwargs)
    else:
        assert_training_model(model)

    # compute the anchors
    #features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5']]
    #locations = __build_locations(features)

    regression = model.outputs[0]
    classification = model.outputs[1]
    centers = model.outputs[2]

    #boxes3D = layers.RegressBoxes3D(name='boxes3D')([regression, locations])
    boxes3D = regression

    # construct the model
    #return keras.models.Model(inputs=model.inputs, outputs=[boxes3D, classification], name=name)
    return keras.models.Model(inputs=model.inputs, outputs=[boxes3D, classification, centers], name=name)
