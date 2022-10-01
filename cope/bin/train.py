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
from tensorflow.keras import mixed_precision
import tensorflow.keras.preprocessing.image
import tensorflow as tf
import json
from tensorflow.python.framework.ops import disable_eager_execution

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import cope.bin  # noqa: F401
    __package__ = "cope.bin"

from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel
#from ..callbacks.eval import Evaluate
from ..models.model import inference_model
from ..utils.anchors import make_shapes_callback
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator
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


def create_models(backbone_model, num_classes, obj_correspondences, obj_diameters, intrinsics, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5):

    modifier = freeze_model if freeze_backbone else None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    # deprecated, not tested with tensorflow [ST]
    if multi_gpu > 1:
        from tensorflow.keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_model(num_classes=num_classes, correspondences=obj_correspondences, obj_diameters=obj_diameters, intrinsics=intrinsics, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_model(num_classes=num_classes, correspondences=obj_correspondences, obj_diameters=obj_diameters, intrinsics=intrinsics, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    custom_model = CustomModel(model=training_model)

    # compile model
    custom_model.compile(
        loss={
            'pts'           : losses.per_cls_l1_sym(num_classes=num_classes, weight=1.3),
            'box'           : losses.per_cls_l1(num_classes=num_classes, weight=1.0),
            'cls'           : losses.per_cls_focal(num_classes=num_classes, weight=1.2),
            'tra'           : losses.per_cls_l1_trans(num_classes=num_classes, weight=1.0),
            'rot'           : losses.per_cls_l1_sym(num_classes=num_classes, weight=0.3),
            'con'           : losses.projection_deviation(num_classes=num_classes, weight=0.1),
            'pro'           : losses.per_cls_l1_rep(num_classes=num_classes, weight=0.15),
        },
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=0.001)
    )

    return model, custom_model


def create_callbacks(model, args, validation_generator=None, train_generator=None):
    callbacks = []

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                'cope_{dataset_type}_{{epoch:02d}}.h5'.format(dataset_type=args.dataset_type)
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


def create_generators(args, preprocess_image):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    '''
    common_args = {
        'batch_size'       : args.batch_size,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'preprocess_image' : preprocess_image,
    }

    transform_generator = random_transform_generator(
            min_translation=(-0.2, -0.2),
            max_translation=(0.2, 0.2),
            min_scaling=(0.8, 0.8),
            max_scaling=(1.2, 1.2),
        )
    '''

    if args.dataset_type == 'linemod':
        from ..preprocessing.data_linemod import LinemodDataset

        dataset = LinemodDataset(args.linemod_path, 'train', batch_size=args.batch_size)
        num_classes = 15
        train_samples = 50000
        dataset = tf.data.Dataset.range(args.workers).interleave(
            lambda _: dataset,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE
            num_parallel_calls=args.workers
        )
        dataset = dataset.shuffle(1, reshuffle_each_iteration=True)
        mesh_info = os.path.join(args.linemod_path, 'annotations', 'models_info' + '.json')
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
            #x_plus = (value['size_x'] / norm_pts) * (value['diameter'] * 0.5)
            #y_plus = (value['size_y'] / norm_pts) * (value['diameter'] * 0.5)
            #z_plus = (value['size_z'] / norm_pts) * (value['diameter'] * 0.5)
            #x_minus = x_plus * -1.0
            #y_minus = y_plus * -1.0
            #z_minus = z_plus * -1.0
            three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                       [x_plus, y_plus, z_minus],
                                       [x_plus, y_minus, z_minus],
                                       [x_plus, y_minus, z_plus],
                                       [x_minus, y_plus, z_plus],
                                       [x_minus, y_plus, z_minus],
                                       [x_minus, y_minus, z_minus],
                                       [x_minus, y_minus, z_plus]])
            correspondences[int(key)-1, :, :] = three_box_solo
            #sphere_diameters[int(key)-1] = value['diameter']
            sphere_diameters[int(key)-1] = norm_pts
        path = os.path.join(args.linemod_path, 'annotations', 'instances_train.json')
        with open(path, 'r') as js:
            data = json.load(js)
        image_ann = data["images"]
        intrinsics = np.ndarray((4), dtype=np.float32)
        for img in image_ann:
            if "fx" in img:
                intrinsics[0] = img["fx"]
                intrinsics[1] = img["fy"]
                intrinsics[2] = img["cx"]
                intrinsics[3] = img["cy"]
            break

    elif args.dataset_type == 'occlusion':
        from ..preprocessing.data_occlusion import OcclusionDataset
        dataset = OcclusionDataset(args.occlusion_path, 'train', batch_size=args.batch_size)
        num_classes = 8
        train_samples = 50000
        dataset = tf.data.Dataset.range(args.workers).interleave(
            lambda _: dataset,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE
            num_parallel_calls=args.workers
        )
        dataset = dataset.shuffle(1, reshuffle_each_iteration=True)
        mesh_info = os.path.join(args.occlusion_path, 'annotations', 'models_info' + '.json')
        correspondences = np.ndarray((num_classes, 8, 3), dtype=np.float32)
        sphere_diameters = np.ndarray((num_classes), dtype=np.float32)
        inv_clss = {1: 0, 5: 1, 6: 2, 8: 3, 9: 4, 10: 5, 11: 6, 12: 7}
        for key, value in json.load(open(mesh_info)).items():
            if int(key) not in [1, 5, 6, 8, 9, 10, 11, 12]:
                continue
            inv_cls = inv_clss[int(key)]
            x_minus = value['min_x']
            y_minus = value['min_y']
            z_minus = value['min_z']
            x_plus = value['size_x'] + x_minus
            y_plus = value['size_y'] + y_minus
            z_plus = value['size_z'] + z_minus
            norm_pts = np.linalg.norm(np.array([value['size_x'], value['size_y'], value['size_z']]))
            # x_plus = (value['size_x'] / norm_pts) * (value['diameter'] * 0.5)
            # y_plus = (value['size_y'] / norm_pts) * (value['diameter'] * 0.5)
            # z_plus = (value['size_z'] / norm_pts) * (value['diameter'] * 0.5)
            # x_minus = x_plus * -1.0
            # y_minus = y_plus * -1.0
            # z_minus = z_plus * -1.0
            three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                       [x_plus, y_plus, z_minus],
                                       [x_plus, y_minus, z_minus],
                                       [x_plus, y_minus, z_plus],
                                       [x_minus, y_plus, z_plus],
                                       [x_minus, y_plus, z_minus],
                                       [x_minus, y_minus, z_minus],
                                       [x_minus, y_minus, z_plus]])
            correspondences[inv_cls, :, :] = three_box_solo
            # sphere_diameters[int(key)-1] = value['diameter']
            sphere_diameters[inv_cls] = norm_pts
        path = os.path.join(args.occlusion_path, 'annotations', 'instances_train.json')
        with open(path, 'r') as js:
            data = json.load(js)
        image_ann = data["images"]
        intrinsics = np.ndarray((4), dtype=np.float32)
        for img in image_ann:
            if "fx" in img:
                intrinsics[0] = img["fx"]
                intrinsics[1] = img["fy"]
                intrinsics[2] = img["cx"]
                intrinsics[3] = img["cy"]
            break

    elif args.dataset_type == 'ycbv':
        from ..preprocessing.data_ycbv import YcbvDataset

        dataset = YcbvDataset(args.ycbv_path, 'train', batch_size=args.batch_size)
        num_classes = 21
        train_samples = 50000
        dataset = tf.data.Dataset.range(args.workers).interleave(
            lambda _: dataset,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE
            num_parallel_calls=args.workers
        )
        mesh_info = os.path.join(args.ycbv_path, 'annotations', 'models_info' + '.yml')
        correspondences = np.ndarray((num_classes, 8, 3), dtype=np.float32)
        sphere_diameters = np.ndarray((num_classes), dtype=np.float32)
        for key, value in yaml.load(open(mesh_info)).items():
            x_minus = value['min_x']
            y_minus = value['min_y']
            z_minus = value['min_z']
            x_plus = value['size_x'] + x_minus
            y_plus = value['size_y'] + y_minus
            z_plus = value['size_z'] + z_minus
            norm_pts = np.linalg.norm(np.array([value['size_x'], value['size_y'], value['size_z']]))
            #x_plus = (value['size_x'] / norm_pts) * (value['diameter'] * 0.5)
            #y_plus = (value['size_y'] / norm_pts) * (value['diameter'] * 0.5)
            #z_plus = (value['size_z'] / norm_pts) * (value['diameter'] * 0.5)
            #x_minus = x_plus * -1.0
            #y_minus = y_plus * -1.0
            #z_minus = z_plus * -1.0
            three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                       [x_plus, y_plus, z_minus],
                                       [x_plus, y_minus, z_minus],
                                       [x_plus, y_minus, z_plus],
                                       [x_minus, y_plus, z_plus],
                                       [x_minus, y_plus, z_minus],
                                       [x_minus, y_minus, z_minus],
                                       [x_minus, y_minus, z_plus]])
            correspondences[int(key)-1, :, :] = three_box_solo
            #sphere_diameters[int(key)-1] = value['diameter']
            sphere_diameters[int(key)-1] = norm_pts
        path = os.path.join(args.ycbv_path, 'annotations', 'instances_train.json')
        with open(path, 'r') as js:
            data = json.load(js)
        image_ann = data["images"]
        intrinsics = np.ndarray((4), dtype=np.float32)
        for img in image_ann:
            if "fx" in img:
                intrinsics[0] = img["fx"]
                intrinsics[1] = img["fy"]
                intrinsics[2] = img["cx"]
                intrinsics[3] = img["cy"]
            break

    elif args.dataset_type == 'tless':
        from ..preprocessing.data_tless import TlessDataset

        dataset = TlessDataset(args.tless_path, 'train', batch_size=args.batch_size)
        num_classes = 30
        train_samples = 50000
        dataset = tf.data.Dataset.range(args.workers).interleave(
            lambda _: dataset,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE
            num_parallel_calls=args.workers
        )
        dataset = dataset.shuffle(1, reshuffle_each_iteration=True)
        mesh_info = os.path.join(args.tless_path, 'annotations', 'models_info' + '.json')
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
            #x_plus = (value['size_x'] / norm_pts) * (value['diameter'] * 0.5)
            #y_plus = (value['size_y'] / norm_pts) * (value['diameter'] * 0.5)
            #z_plus = (value['size_z'] / norm_pts) * (value['diameter'] * 0.5)
            #x_minus = x_plus * -1.0
            #y_minus = y_plus * -1.0
            #z_minus = z_plus * -1.0
            three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                       [x_plus, y_plus, z_minus],
                                       [x_plus, y_minus, z_minus],
                                       [x_plus, y_minus, z_plus],
                                       [x_minus, y_plus, z_plus],
                                       [x_minus, y_plus, z_minus],
                                       [x_minus, y_minus, z_minus],
                                       [x_minus, y_minus, z_plus]])
            correspondences[int(key)-1, :, :] = three_box_solo
            #sphere_diameters[int(key)-1] = value['diameter']
            sphere_diameters[int(key)-1] = norm_pts
        path = os.path.join(args.tless_path, 'annotations', 'instances_train.json')
        with open(path, 'r') as js:
            data = json.load(js)
        image_ann = data["images"]
        intrinsics = np.ndarray((4), dtype=np.float32)
        for img in image_ann:
            if "fx" in img:
                intrinsics[0] = img["fx"]
                intrinsics[1] = img["fy"]
                intrinsics[2] = img["cx"]
                intrinsics[3] = img["cy"]
            break
        intrinsics *= 1.0/1.125

    elif args.dataset_type == 'homebrewed':
        from ..preprocessing.data_hb import HomebrewedDataset

        dataset = HomebrewedDataset(args.hb_path, 'train', batch_size=args.batch_size)
        num_classes = 33
        train_samples = 50000
        dataset = tf.data.Dataset.range(args.workers).interleave(
            lambda _: dataset,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE
            num_parallel_calls=args.workers
        )
        dataset = dataset.shuffle(1, reshuffle_each_iteration=True)
        mesh_info = os.path.join(args.hb_path, 'annotations', 'models_info' + '.json')
        correspondences = np.ndarray((num_classes, 8, 3), dtype=np.float32)
        sphere_diameters = np.ndarray((num_classes), dtype=np.float32)
        for key, value in yaml.load(open(mesh_info)).items():
            x_minus = value['min_x']
            y_minus = value['min_y']
            z_minus = value['min_z']
            x_plus = value['size_x'] + x_minus
            y_plus = value['size_y'] + y_minus
            z_plus = value['size_z'] + z_minus
            norm_pts = np.linalg.norm(np.array([value['size_x'], value['size_y'], value['size_z']]))
            #x_plus = (value['size_x'] / norm_pts) * (value['diameter'] * 0.5)
            #y_plus = (value['size_y'] / norm_pts) * (value['diameter'] * 0.5)
            #z_plus = (value['size_z'] / norm_pts) * (value['diameter'] * 0.5)
            #x_minus = x_plus * -1.0
            #y_minus = y_plus * -1.0
            #z_minus = z_plus * -1.0
            three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                       [x_plus, y_plus, z_minus],
                                       [x_plus, y_minus, z_minus],
                                       [x_plus, y_minus, z_plus],
                                       [x_minus, y_plus, z_plus],
                                       [x_minus, y_plus, z_minus],
                                       [x_minus, y_minus, z_minus],
                                       [x_minus, y_minus, z_plus]])
            correspondences[int(key)-1, :, :] = three_box_solo
            #sphere_diameters[int(key)-1] = value['diameter']
            sphere_diameters[int(key)-1] = norm_pts
        path = os.path.join(args.hb_path, 'annotations', 'instances_train.json')
        with open(path, 'r') as js:
            data = json.load(js)
        image_ann = data["images"]
        intrinsics = np.ndarray((4), dtype=np.float32)
        for img in image_ann:
            if "fx" in img:
                intrinsics[0] = img["fx"]
                intrinsics[1] = img["fy"]
                intrinsics[2] = img["cx"]
                intrinsics[3] = img["cy"]
            break

    elif args.dataset_type == 'icbin':
        from ..preprocessing.data_icbin import ICbinDataset

        dataset = ICbinDataset(args.icbin_path, 'train', batch_size=args.batch_size)
        num_classes = 2
        train_samples = 50000
        dataset = tf.data.Dataset.range(args.workers).interleave(
            lambda _: dataset,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE
            num_parallel_calls=args.workers
        )
        dataset = dataset.shuffle(1, reshuffle_each_iteration=True)
        mesh_info = os.path.join(args.icbin_path, 'annotations', 'models_info' + '.json')
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
            #x_plus = (value['size_x'] / norm_pts) * (value['diameter'] * 0.5)
            #y_plus = (value['size_y'] / norm_pts) * (value['diameter'] * 0.5)
            #z_plus = (value['size_z'] / norm_pts) * (value['diameter'] * 0.5)
            #x_minus = x_plus * -1.0
            #y_minus = y_plus * -1.0
            #z_minus = z_plus * -1.0
            three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                       [x_plus, y_plus, z_minus],
                                       [x_plus, y_minus, z_minus],
                                       [x_plus, y_minus, z_plus],
                                       [x_minus, y_plus, z_plus],
                                       [x_minus, y_plus, z_minus],
                                       [x_minus, y_minus, z_minus],
                                       [x_minus, y_minus, z_plus]])
            correspondences[int(key)-1, :, :] = three_box_solo
            #sphere_diameters[int(key)-1] = value['diameter']
            sphere_diameters[int(key) - 1] = norm_pts
        path = os.path.join(args.icbin_path, 'annotations', 'instances_train.json')
        with open(path, 'r') as js:
            data = json.load(js)
        image_ann = data["images"]
        intrinsics = np.ndarray((4), dtype=np.float32)
        for img in image_ann:
            if "fx" in img:
                intrinsics[0] = img["fx"]
                intrinsics[1] = img["fy"]
                intrinsics[2] = img["cx"]
                intrinsics[3] = img["cy"]
            break

    elif args.dataset_type == 'custom':
        from ..preprocessing.data_custom import CustomDataset

        dataset = CustomDataset(args.custom_path, 'train', batch_size=args.batch_size)
        num_classes = 7
        #train_samples = 624
        train_samples = 50000
        dataset = tf.data.Dataset.range(args.workers).interleave(
            lambda _: dataset,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE
            num_parallel_calls=args.workers
        )
        mesh_info = os.path.join(args.custom_path, 'annotations', 'models_info' + '.json')
        correspondences = np.ndarray((num_classes, 8, 3), dtype=np.float32)
        sphere_diameters = np.ndarray((num_classes), dtype=np.float32)
        for key, value in json.load(open(mesh_info)).items():
            if int(key) > 6:
                key = int(key) - 1
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
            #sphere_diameters[int(key)-1] = value['diameter']
            sphere_diameters[int(key) - 1] = norm_pts
        path = os.path.join(args.custom_path, 'annotations', 'instances_train.json')
        with open(path, 'r') as js:
            data = json.load(js)
        image_ann = data["images"]
        intrinsics = np.ndarray((4), dtype=np.float32)
        for img in image_ann:
            if "fx" in img:
                intrinsics[0] = img["fx"]
                intrinsics[1] = img["fy"]
                intrinsics[2] = img["cx"]
                intrinsics[3] = img["cy"]
            break

    elif args.dataset_type == 'rost':
        from ..preprocessing.data_rost import RostDataset

        dataset = RostDataset(args.rost_path, 'train', batch_size=args.batch_size)
        num_classes = 6
        train_samples = 1664 * 8 * 3
        dataset = tf.data.Dataset.range(args.workers).interleave(
            lambda _: dataset,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE
            num_parallel_calls=args.workers
        )
        mesh_info = os.path.join(args.rost_path, 'annotations', 'models_info' + '.json')
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
            #sphere_diameters[int(key)-1] = value['diameter']
            sphere_diameters[int(key) - 1] = norm_pts
        path = os.path.join(args.rost_path, 'annotations', 'instances_train.json')
        with open(path, 'r') as js:
            data = json.load(js)
        image_ann = data["images"]
        intrinsics = np.ndarray((4), dtype=np.float32)
        for img in image_ann:
            if "fx" in img:
                intrinsics[0] = img["fx"]
                intrinsics[1] = img["fy"]
                intrinsics[2] = img["cx"]
                intrinsics[3] = img["cy"]
            break

    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return dataset, num_classes, correspondences, sphere_diameters, train_samples, intrinsics


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network with object pose estimation.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    linemod_parser = subparsers.add_parser('linemod')
    linemod_parser.add_argument('linemod_path', help='Path to dataset directory (ie. /tmp/linemod).')

    occlusion_parser = subparsers.add_parser('occlusion')
    occlusion_parser.add_argument('occlusion_path', help='Path to dataset directory (ie. /tmp/linemod).')

    ycbv_parser = subparsers.add_parser('ycbv')
    ycbv_parser.add_argument('ycbv_path', help='Path to dataset directory (ie. /tmp/ycbv).')

    tless_parser = subparsers.add_parser('tless')
    tless_parser.add_argument('tless_path', help='Path to dataset directory (ie. /tmp/tless).')

    hb_parser = subparsers.add_parser('homebrewed')
    hb_parser.add_argument('hb_path', help='Path to dataset directory (ie. /tmp/hb).')

    icbin_parser = subparsers.add_parser('icbin')
    icbin_parser.add_argument('icbin_path', help='Path to dataset directory (ie. /tmp/icbin).')

    custom_parser = subparsers.add_parser('custom')
    custom_parser.add_argument('custom_path', help='Path to dataset directory (ie. /tmp/custom).')

    rost_parser = subparsers.add_parser('rost')
    rost_parser.add_argument('rost_path', help='Path to dataset directory (ie. /tmp/custom).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

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

    #disable_eager_execution()

    #backbone = models.backbone('resnet50')
    backbone = models.backbone('resnet101')
    #backbone = models.backbone('resnet152')
    #backbone = models.backbone('efficientnet')
    #backbone = models.backbone('darknet')
    #backbone = models.backbone('xception')
    #backbone = models.backbone('nasnetmobile')

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # create the generators
    dataset, num_classes, correspondences, obj_diameters, train_samples, intrinsics = create_generators(args, backbone.preprocess_image)

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
        prediction_model = inference_model(model=model)
    else:
        weights = args.weights

        print('Creating model, this may take a second...')
        model, training_model = create_models(
            backbone_model=backbone.model,
            num_classes=num_classes,
            obj_correspondences=correspondences,
            obj_diameters=obj_diameters,
            intrinsics=intrinsics,
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
        #steps_per_epoch=10,
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

