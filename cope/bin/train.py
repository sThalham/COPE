#!/usr/bin/env python

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

import argparse
import os
import sys
import numpy as np

import tensorflow.keras as keras
import tensorflow as tf
import json

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import cope.bin  # noqa: F401
    __package__ = "cope.bin"

from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..utils.model import freeze as freeze_model
from ..models.train_step import CustomModel


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_model, num_classes, obj_correspondences, obj_diameters, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5):

    modifier = freeze_model if freeze_backbone else None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    # deprecated, not tested with tensorflow [ST]
    if multi_gpu > 1:
        from tensorflow.keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_model(num_classes=num_classes, correspondences=obj_correspondences, obj_diameters=obj_diameters, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_model(num_classes=num_classes, correspondences=obj_correspondences, obj_diameters=obj_diameters, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    custom_model = CustomModel(model=training_model)

    # compile model
    custom_model.compile(
        loss={
            'pts'           : losses.per_cls_l1_sym(num_classes=num_classes, weight=1.3),
            'box'           : losses.per_cls_l1(num_classes=num_classes, weight=1.0),
            'cls'           : losses.focal(),
            'tra'           : losses.per_cls_l1_trans(num_classes=num_classes, weight=1.0),
            'rot'           : losses.per_cls_l1_sym(num_classes=num_classes, weight=0.3),
            'con'           : losses.projection_deviation(num_classes=num_classes, weight=0.1),
            'pro'           : losses.per_cls_l1_rep(num_classes=num_classes, weight=0.15),
        },
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=0.001)
    )

    return model, custom_model


def create_callbacks(model, args):
    callbacks = []

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                'cope_{dataset_type}_{{epoch:02d}}.h5'.format(dataset_type=args.dataset)
            ),
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))

    return callbacks


def create_generators(args):

    from ..preprocessing.data_generator import GeneratorDataset

    mesh_info = os.path.join(args.data_path, 'meshes', 'models_info' + '.json')
    #num_classes = len(json.load(open(mesh_info)).items())
    num_classes = 21
    train_samples = 162153
    dataset = GeneratorDataset(args.data_path, 'train', num_classes=num_classes, batch_size=args.batch_size)
    dataset = tf.data.Dataset.range(args.workers).interleave(
        lambda _: dataset,
        # num_parallel_calls=tf.data.experimental.AUTOTUNE
        num_parallel_calls=args.workers
    )
    dataset = dataset.shuffle(1, reshuffle_each_iteration=True)
    correspondences = np.ndarray((num_classes, 8, 3), dtype=np.float32)
    sphere_diameters = np.ndarray((num_classes), dtype=np.float32)

    for key, value in json.load(open(mesh_info)).items():
        x_minus = value['min_x']
        y_minus = value['min_y']
        z_minus = value['min_z']
        x_plus = value['size_x'] + x_minus
        y_plus = value['size_y'] + y_minus
        z_plus = value['size_z'] + z_minus
        norm_pts = np.linalg.norm(np.array([value['size_x'], value['size_y'], value['size_z']]))
        three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                   [x_plus, y_plus, z_minus],
                                   [x_plus, y_minus, z_minus],
                                   [x_plus, y_minus, z_plus],
                                   [x_minus, y_plus, z_plus],
                                   [x_minus, y_plus, z_minus],
                                   [x_minus, y_minus, z_minus],
                                   [x_minus, y_minus, z_plus]])
        correspondences[int(key)-1, :, :] = three_box_solo
        sphere_diameters[int(key)-1] = norm_pts

    return dataset, num_classes, correspondences, sphere_diameters, train_samples


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network with object pose estimation.')
    #subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    #subparsers.required = False

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('dataset',             help='Path to dataset directory (ie. /tmp/your_converted_dataset).')
    parser.add_argument('--data-path',           help='Path to dataset directory (ie. /tmp/your_converted_dataset).')
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=100)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./models\')', default='./models')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=480)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=640)

    # Fit generator arguments
    parser.add_argument('--workers', help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0', type=int, default=1)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int, default=1)

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    #backbone = models.backbone('resnet50')
    backbone = models.backbone('resnet101')

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # create the generators
    dataset, num_classes, correspondences, obj_diameters, train_samples = create_generators(args)

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
    else:
        weights = args.weights

        print('Creating model, this may take a second...')
        model, training_model = create_models(
            backbone_model=backbone.model,
            num_classes=num_classes,
            obj_correspondences=correspondences,
            obj_diameters=obj_diameters,
            weights=weights,
            multi_gpu=0,
            freeze_backbone=args.freeze_backbone,
            lr=args.lr,
        )

    # print model summary
    print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(
        model,
        args,
    )

    # Use multiprocessing if workers > 0
    if args.workers > 0:
        use_multiprocessing = True
    else:
        use_multiprocessing = False

    training_model.fit(
        x=dataset,
        steps_per_epoch=train_samples / args.batch_size,
        epochs=args.epochs,
        #epochs=1,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=args.max_queue_size
    )


if __name__ == '__main__':
    main()

