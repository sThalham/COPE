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
        #'kernel_regularizer': keras.regularizers.l2(0.001),
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

    #cent = keras.layers.Conv2D(
    #    filters=1,
    #    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
    #    bias_initializer=initializers.PriorProbability(probability=prior_probability),
    #    **options
    #)(outputs)

    # reshape output and apply sigmoid
    #if keras.backend.image_data_format() == 'channels_first':
    #    cent = keras.layers.Permute((2, 3, 1))(cent)
    #cent = keras.layers.Reshape((-1, 1))(cent) # centerness = keras.layers.Reshape((-1, num_classes))(centerness)
    #cent = keras.layers.Activation('sigmoid')(cent)

    return keras.models.Model(inputs=inputs, outputs=labels)#, keras.models.Model(inputs=inputs, outputs=cent)


def default_boxes_model(num_values, pyramid_feature_size=256, prior_probability=0.01, regression_feature_size=256):
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros',
        #'kernel_regularizer' : keras.regularizers.l2(0.001),
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

    regress = keras.layers.Conv2D(num_values, **options)(outputs)
    #regress = keras.layers.SeparableConv2D(num_values, **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        regress = keras.layers.Permute((2, 3, 1))(regress)
    regress = keras.layers.Reshape((-1, num_values))(regress)

    return keras.models.Model(inputs=inputs, outputs=regress)


def default_regression_model(num_values, pyramid_feature_size=256, prior_probability=0.01, regression_feature_size=512):
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros',
        'kernel_regularizer' : keras.regularizers.l2(0.001),
        #'pointwise_regularizer': keras.regularizers.l2(0.001),
        #'depthwise_regularizer': keras.regularizers.l2(0.001),
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

    regress = keras.layers.Conv2D(num_values, **options)(outputs)
    #regress = keras.layers.SeparableConv2D(num_values, **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        regress = keras.layers.Permute((2, 3, 1))(regress)
    regress = keras.layers.Reshape((-1, num_values))(regress)

    #conf = keras.layers.Conv2D(
    #    filters=num_values,
    #    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
    #    bias_initializer=initializers.PriorProbability(probability=prior_probability),
    #    **options_centerness
    #)(outputs)

    # reshape output and apply sigmoid
    #if keras.backend.image_data_format() == 'channels_first':
    #    conf = keras.layers.Permute((2, 3, 1))(conf)
    #conf = keras.layers.Reshape((-1, num_values))(conf) # centerness = keras.layers.Reshape((-1, num_classes))(centerness)
    #conf = keras.layers.Activation('sigmoid')(conf)

    #return keras.models.Model(inputs=inputs, outputs=regress)
    return keras.models.Model(inputs=inputs, outputs=regress)#, keras.models.Model(inputs=inputs, outputs=conf)


def __create_PFPN(C3, C4, C5, feature_size=256):
    options = {
        #'kernel_regularizer': keras.regularizers.l2(0.001),
    }
    
    # 3x3 conv for test 4
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', **options)(C3)
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', **options)(C4)
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', **options)(C5)

    P5_upsampled = layers.UpsampleLike()([P5, P4])
    P4_upsampled = layers.UpsampleLike()([P4, P3])
    P4_mid = keras.layers.Add()([P5_upsampled, P4])
    P4_mid = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, activation='linear', padding='same', **options)(P4_mid)
    P3_mid = keras.layers.Add()([P4_upsampled, P3])
    P3_mid = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, activation='linear', padding='same', **options)(P3_mid)
    P3_down = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, activation='linear', padding='same', **options)(P3_mid)
    P3_fin = keras.layers.Add()([P3_mid, P3])  # skip connection
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, activation='linear', padding='same', name='P3', **options)(P3_fin)

    P4_fin = keras.layers.Add()([P3_down, P4_mid])
    P4_down = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, activation='linear', padding='same', **options)(P4_mid)
    P4_fin = keras.layers.Add()([P4_fin, P4])  # skip connection
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, activation='linear', padding='same', name='P4', **options)(P4_fin)
    P5_fin = keras.layers.Add()([P4_down, P5])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, activation='linear', padding='same', name='P5', **options)(P5_fin)

    return [P3, P4, P5]


def pyrapose(
    inputs,
    backbone_layers,
    num_classes,
    create_pyramid_features = __create_PFPN,
    name                    = 'pyrapose'
):
    regression_branch = default_regression_model(16)
    #boxes_branch = default_regression_model(4)
    location_branch = default_classification_model(num_classes)

    b1, b2, b3 = backbone_layers
    P3, P4, P5 = create_pyramid_features(b1, b2, b3)

    pyramids = []
    regression_P3 = regression_branch(P3)
    regression_P4 = regression_branch(P4)
    regression_P5 = regression_branch(P5)
    regression = keras.layers.Concatenate(axis=1, name='points')([regression_P3, regression_P4, regression_P5])
    pyramids.append(regression)

    #boxes_P3 = boxes_branch(P3)
    #boxes_P4 = boxes_branch(P4)
    #boxes_P5 = boxes_branch(P5)
    #boxes = keras.layers.Concatenate(axis=1, name='boxes')([boxes_P3, boxes_P4, boxes_P5])
    #pyramids.append(boxes)

    #residuals_P3 = regression_branch[1](P3)
    #residuals_P4 = regression_branch[1](P4)
    #residuals_P5 = regression_branch[1](P5)
    #residuals = keras.layers.Concatenate(axis=1)([residuals_P3, residuals_P4, residuals_P5])
    #residual_predictions = keras.layers.Concatenate(axis=2, name='res')([regression, residuals])
    #pyramids.append(residual_predictions)

    location_P3 = location_branch(P3)
    location_P4 = location_branch(P4)
    location_P5 = location_branch(P5)
    pyramids.append(keras.layers.Concatenate(axis=1, name='cls')([location_P3, location_P4, location_P5]))

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def __build_locations(features):
    strides = [8, 16, 32]
    locations = [
        layers.Locations(stride=strides[i], name='locations_{}'.format(i))(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='locations')(locations)


def inference_model(
    model                 = None,
    object_diameters      = None,
    name                  = 'pyrapose',
    score_threshold       = 0.5,
    max_detections        = 100,
    **kwargs
):

    # create RetinaNet model
    if model is None:
        model = pyrapose(**kwargs)
    else:
        assert_training_model(model)

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5']]
    locations = __build_locations(features)

    regression = model.outputs[0]
    #residuals = model.outputs[1][:, :, 18:]
    classification = model.outputs[1]

    detections = layers.FilterDetections(
        name='filtered_detections',
        score_threshold=score_threshold,
        max_detections=max_detections,
    )([regression, classification, locations])

    print(detections[2])
    tf_diameter = tf.convert_to_tensor(object_diameters)
    #print(tf_diameter)
    #rep_object_diameters = tf.zeros(max_detections)
    #print(rep_object_diameters)
    #rep_object_diameters = tf_diameter[detections[2]]

    #c = lambda i: tf.less(i, max_detections+1)
    #b = lambda i: detections[2][i]
    #rep_object_diameters = tf.while_loop(
    #    c, b, loop_vars=[detections[2], tf_diameter])

    rep_object_diameters = tf.gather(tf_diameter,
                indices=detections[3])

    boxes3D = layers.RegressBoxes3D(name='boxes3D')([detections[0], detections[1], rep_object_diameters])

    # construct the model
    #return keras.models.Model(inputs=model.inputs, outputs=[boxes3D, classification], name=name)
    #return keras.models.Model(inputs=model.inputs, outputs=[regression, classification], name=name)
    return keras.models.Model(inputs=model.inputs, outputs=[boxes3D, detections[2], detections[3]], name=name)