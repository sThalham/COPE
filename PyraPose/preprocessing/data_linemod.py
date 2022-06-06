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

from ..utils.image import read_image_bgr
from collections import defaultdict

import numpy as np
import os
import json
import yaml
import itertools
import random
import warnings
import copy
import cv2
import time
import imgaug.augmenters as iaa

import tensorflow.keras as keras
import tensorflow as tf

from ..utils.anchors import (
    anchor_targets_bbox,
    guess_shapes
)
from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    adjust_transform_for_mask,
    apply_transform2mask,
    adjust_pose_annotation,
    apply_transform,
    preprocess_image,
    resize_image,
    read_image_bgr,
    augment_image
)
from ..utils.transform import transform_aabb, random_transform_generator


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def load_classes(categories):
    """ Loads the class to label mapping (and inverse) for COCO.
    """
    if _isArrayLike(categories):
        categories = [categories[id] for id in categories]
    elif type(categories) == int:
        categories = [categories[categories]]
    categories.sort(key=lambda x: x['id'])
    classes = {}
    labels = {}
    labels_inverse = {}
    for c in categories:
        labels[len(classes)] = c['id']
        labels_inverse[c['id']] = len(classes)
        classes[c['name']] = len(classes)
    # also load the reverse (label -> name)
    labels_rev = {}
    for key, value in classes.items():
        labels_rev[value] = key

    return classes, labels, labels_inverse, labels_rev


class LinemodDataset(tf.data.Dataset):

    def _sample(data_dir, set_name, batch_size, image_min_side=480, image_max_side=640):

        def _isArrayLike(obj):
            return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

        # Parameters
        data_dir = data_dir.decode("utf-8")
        set_name = set_name.decode("utf-8")
        batch_size = batch_size
        path = os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json')
        mesh_info = os.path.join(data_dir, 'annotations', 'models_info' + '.json')

        with open(path, 'r') as js:
            data = json.load(js)

        # load source w/ annotations
        image_ann = data["images"]
        anno_ann = data["annotations"]
        cat_ann = data["categories"]
        cats = {}
        image_ids = []
        image_paths = []
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)

        for img in image_ann:
            if "fx" in img:
                fx = img["fx"]
                fy = img["fy"]
                cx = img["cx"]
                cy = img["cy"]
            image_ids.append(img['id'])  # to correlate indexing to self.image_ann
            image_paths.append(os.path.join(data_dir, 'images', set_name, img['file_name']))

        for cat in cat_ann:
            cats[cat['id']] = cat

        for ann in anno_ann:
            imgToAnns[ann['image_id']].append(ann)
            catToImgs[ann['category_id']].append(ann['image_id'])

        classes, labels, labels_inverse, labels_rev = load_classes(cats)
        num_classes = len(classes)

        # load 3D boxes
        TDboxes = np.ndarray((num_classes + 1, 8, 3), dtype=np.float32)
        sphere_diameters = np.ndarray((num_classes + 1), dtype=np.float32)
        sym_cont = np.zeros((num_classes + 1, 2, 3), dtype=np.float32)
        sym_disc = np.zeros((num_classes + 1, 8, 16), dtype=np.float32)

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
            TDboxes[int(key), :, :] = three_box_solo
            #sphere_diameters[int(key)] = value['diameter']
            sphere_diameters[int(key)] = norm_pts

            if 'symmetries_discrete' in value:
                for sdx, sym in enumerate(value['symmetries_discrete']):
                    sym_disc[int(key), sdx, :] = np.array(sym)
                    sym_disc[int(key), sdx, [3, 7, 11]] *= 0.001
            #else:
                #sym_disc[int(key), :, :] = np.repeat(np.eye((4)).reshape(16)[np.newaxis, :], repeats=3, axis=0)  # np.zeros((3, 16))

            if "symmetries_continuous" in value:
                sym_cont[int(key), 0, :] = np.array(value['symmetries_continuous'][0]['axis'], dtype=np.float32)
                sym_cont[int(key), 1, :] = np.array(value['symmetries_continuous'][0]['offset'], dtype=np.float32)
            #else:
            #    sym_cont[int(key), :, :] = np.zeros((2, 3))

        def load_image(image_index):
            """ Load an image at the image_index.
            """
            path = image_paths[image_index]
            path = path[:-4] + '_rgb' + path[-4:]

            return read_image_bgr(path)

        def load_annotations(image_index):
            """ Load annotations for an image_index.
                CHECK DONE HERE: Annotations + images correct
            """
            ids = image_ids[image_index]
            image_num = int(str(ids)[1:])

            # lists = [imgToAnns[imgId] for imgId in ids if imgId in imgToAnns]
            # anns = list(itertools.chain.from_iterable(lists))
            anns = imgToAnns[image_ids[image_index]]

            path = image_paths[image_index]
            mask_path = path[:-4] + '_mask.png'  # + path[-4:]
            mask = cv2.imread(mask_path, -1)

            #annotations = {'mask': mask, 'labels': np.empty((0,)),
            #               'bboxes': np.empty((0, 4)), 'poses': np.empty((0, 7)), 'segmentations': np.empty((0, 8, 3)), 'diameters': np.empty((0,)),
            #               'cam_params': np.empty((0, 4)), 'mask_ids': np.empty((0,)), 'sym_dis': np.empty((0, 8, 16)), 'sym_con': np.empty((0, 2, 3))}
            annotations = {'image_num': np.array([image_num]),
                            'labels': np.empty((0,)),
                           'bboxes': np.empty((0, 4)),
                           'poses': np.empty((0, 7)),
                           'cam_params': np.empty((0, 4))}

            for idx, a in enumerate(anns):
                annotations['labels'] = np.concatenate([annotations['labels'], [labels_inverse[a['category_id']]]],
                                                       axis=0)
                annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                    a['bbox'][0],
                    a['bbox'][1],
                    a['bbox'][0] + a['bbox'][2],
                    a['bbox'][1] + a['bbox'][3],
                ]]], axis=0)
                if a['pose'][2] < 10.0:  # needed for adjusting pose annotations
                    a['pose'][0] = a['pose'][0] * 1000.0
                    a['pose'][1] = a['pose'][1] * 1000.0
                    a['pose'][2] = a['pose'][2] * 1000.0
                annotations['poses'] = np.concatenate([annotations['poses'], [[
                    a['pose'][0],
                    a['pose'][1],
                    a['pose'][2],
                    a['pose'][3],
                    a['pose'][4],
                    a['pose'][5],
                    a['pose'][6],
                ]]], axis=0)
                #annotations['mask_ids'] = np.concatenate([annotations['mask_ids'], [
                #    a['mask_id'],
                #]], axis=0)
                #objID = a['category_id']
                #threeDbox = TDboxes[objID, :, :]
                #annotations['segmentations'] = np.concatenate([annotations['segmentations'], [threeDbox]], axis=0)
                #annotations['diameters'] = np.concatenate([annotations['diameters'], [sphere_diameters[objID]]],
                #                                          axis=0)
                annotations['cam_params'] = np.concatenate([annotations['cam_params'], [[
                    fx,
                    fy,
                    cx,
                    cy,
                ]]], axis=0)
                #annotations['sym_dis'] = np.concatenate(
                #    [annotations['sym_dis'], sym_disc[objID, :, :][np.newaxis, ...]], axis=0)
                #annotations['sym_con'] = np.concatenate(
                #    [annotations['sym_con'], sym_cont[objID, :, :][np.newaxis, ...]], axis=0)

            return annotations

        for image_index, image_path in enumerate(image_paths):
            x_t = load_image(image_index)
            y_t = load_annotations(image_index)

            x_t, scale = resize_image(x_t, min_side=image_min_side, max_side=image_max_side)
            y_t['bboxes'] *= scale
            y_t['cam_params'] *= scale

            x_t = preprocess_image(x_t)
            x_t = keras.backend.cast_to_floatx(x_t)

            anno = []
            for adx, item in enumerate(y_t.items()):
                anno.append(item[1])

            img_path = image_paths[image_index]
            scene_id = np.array([int(img_path[-15:-9])])

            yield scene_id, anno[0], x_t, anno[1], anno[2], anno[3], anno[4]

    def _generate(data_dir, set_name, batch_size=8, transform_generator=None, image_min_side=480,
                         image_max_side=640):

        def _isArrayLike(obj):
            return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

        def load_classes(categories):
            """ Loads the class to label mapping (and inverse) for COCO.
            """
            if _isArrayLike(categories):
                categories = [categories[id] for id in categories]
            elif type(categories) == int:
                categories = [categories[categories]]
            categories.sort(key=lambda x: x['id'])
            classes = {}
            labels = {}
            labels_inverse = {}
            for c in categories:
                labels[len(classes)] = c['id']
                labels_inverse[c['id']] = len(classes)
                classes[c['name']] = len(classes)
            # also load the reverse (label -> name)
            labels_rev = {}
            for key, value in classes.items():
                labels_rev[value] = key

            return classes, labels, labels_inverse, labels_rev

        # Parameters
        data_dir = data_dir.decode("utf-8")
        set_name = set_name.decode("utf-8")
        batch_size = batch_size
        path = os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json')
        mesh_info = os.path.join(data_dir, 'annotations', 'models_info' + '.json')

        batch_size = int(batch_size)
        image_min_side = image_min_side
        image_max_side = image_max_side
        transform_parameters = TransformParameters()
        compute_anchor_targets = anchor_targets_bbox
        compute_shapes = guess_shapes

        with open(path, 'r') as js:
            data = json.load(js)

        # load source w/ annotations
        image_ann = data["images"]
        anno_ann = data["annotations"]
        cat_ann = data["categories"]
        cats = {}
        image_ids = []
        image_paths = []
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)

        for img in image_ann:
            if "fx" in img:
                fx = img["fx"]
                fy = img["fy"]
                cx = img["cx"]
                cy = img["cy"]
            image_ids.append(img['id'])  # to correlate indexing to self.image_ann
            image_paths.append(os.path.join(data_dir, 'images', set_name, img['file_name']))

        for cat in cat_ann:
            cats[cat['id']] = cat

        for ann in anno_ann:
            imgToAnns[ann['image_id']].append(ann)
            catToImgs[ann['category_id']].append(ann['image_id'])

        classes, labels, labels_inverse, labels_rev = load_classes(cats)
        num_classes = len(classes)

        # load 3D boxes
        TDboxes = np.ndarray((num_classes + 1, 8, 3), dtype=np.float32)
        sphere_diameters = np.ndarray((num_classes + 1), dtype=np.float32)
        sym_cont = np.zeros((num_classes + 1, 2, 3), dtype=np.float32)
        sym_disc = np.zeros((num_classes + 1, 8, 16), dtype=np.float32)

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
            TDboxes[int(key), :, :] = three_box_solo
            #sphere_diameters[int(key)] = value['diameter']
            sphere_diameters[int(key)] = norm_pts

            if 'symmetries_discrete' in value:
                for sdx, sym in enumerate(value['symmetries_discrete']):
                    sym_disc[int(key), sdx, :] = np.array(sym)
            #else:
                #sym_disc[int(key), :, :] = np.repeat(np.eye((4)).reshape(16)[np.newaxis, :], repeats=3, axis=0)  # np.zeros((3, 16))

            if "symmetries_continuous" in value:
                sym_cont[int(key), 0, :] = np.array(value['symmetries_continuous'][0]['axis'], dtype=np.float32)
                sym_cont[int(key), 1, :] = np.array(value['symmetries_continuous'][0]['offset'], dtype=np.float32)
            #else:
            #    sym_cont[int(key), :, :] = np.zeros((2, 3))

        transform_generator = random_transform_generator(
            min_translation=(0.0, 0.0),
            max_translation=(0.0, 0.0),
            min_scaling=(0.95, 0.95),
            max_scaling=(1.05, 1.05),
        )

        def load_image(image_index):
            """ Load an image at the image_index.
            """
            path = image_paths[image_index]
            path = path[:-4] + '_rgb' + path[-4:]

            return read_image_bgr(path)

        def load_annotations(image_index):
            """ Load annotations for an image_index.
                CHECK DONE HERE: Annotations + images correct
            """
            # ids = image_ids[image_index]

            # lists = [imgToAnns[imgId] for imgId in ids if imgId in imgToAnns]
            # anns = list(itertools.chain.from_iterable(lists))
            anns = imgToAnns[image_ids[image_index]]

            path = image_paths[image_index]
            mask_path = path[:-4] + '_mask.png'  # + path[-4:]
            mask = cv2.imread(mask_path, -1)

            annotations = {'mask': mask, 'visibility': np.empty((0,)), 'labels': np.empty((0,)),
                           'bboxes': np.empty((0, 4)), 'poses': np.empty((0, 7)), 'segmentations': np.empty((0, 8, 3)), 'diameters': np.empty((0,)),
                           'cam_params': np.empty((0, 4)), 'mask_ids': np.empty((0,)), 'sym_dis': np.empty((0, 8, 16)), 'sym_con': np.empty((0, 2, 3))}

            for idx, a in enumerate(anns):
                if set_name == 'train':
                    if a['feature_visibility'] < 0.25:
                        continue
                annotations['visibility'] = np.concatenate([annotations['visibility'], [a['feature_visibility']]])
                annotations['labels'] = np.concatenate([annotations['labels'], [labels_inverse[a['category_id']]]],
                                                       axis=0)
                annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                    a['bbox'][0],
                    a['bbox'][1],
                    a['bbox'][0] + a['bbox'][2],
                    a['bbox'][1] + a['bbox'][3],
                ]]], axis=0)
                if a['pose'][2] < 10.0:  # needed for adjusting pose annotations
                    a['pose'][0] = a['pose'][0] * 1000.0
                    a['pose'][1] = a['pose'][1] * 1000.0
                    a['pose'][2] = a['pose'][2] * 1000.0
                annotations['poses'] = np.concatenate([annotations['poses'], [[
                    a['pose'][0],
                    a['pose'][1],
                    a['pose'][2],
                    a['pose'][3],
                    a['pose'][4],
                    a['pose'][5],
                    a['pose'][6],
                ]]], axis=0)
                annotations['mask_ids'] = np.concatenate([annotations['mask_ids'], [
                    a['mask_id'],
                ]], axis=0)
                objID = a['category_id']
                threeDbox = TDboxes[objID, :, :]
                annotations['segmentations'] = np.concatenate([annotations['segmentations'], [threeDbox]], axis=0)
                annotations['diameters'] = np.concatenate([annotations['diameters'], [sphere_diameters[objID]]],
                                                          axis=0)
                annotations['cam_params'] = np.concatenate([annotations['cam_params'], [[
                    fx,
                    fy,
                    cx,
                    cy,
                ]]], axis=0)
                annotations['sym_dis'] = np.concatenate(
                    [annotations['sym_dis'], sym_disc[objID, :, :][np.newaxis, ...]], axis=0)
                annotations['sym_con'] = np.concatenate(
                    [annotations['sym_con'], sym_cont[objID, :, :][np.newaxis, ...]], axis=0)

            return annotations

        def random_transform_group_entry(image, annotations, transform=None):
            """ Randomly transforms image and annotation.
            """
            # randomly transform both image and annotations
            if transform is not None or transform_generator:
                if transform is None:
                    next_transform = next(transform_generator)
                    transform = adjust_transform_for_image(next_transform, image,
                                                           transform_parameters.relative_translation)
                    transform_mask = adjust_transform_for_mask(next_transform, annotations['mask'],
                                                               transform_parameters.relative_translation)

                image = apply_transform(transform, image, transform_parameters)
                annotations['mask'] = apply_transform2mask(transform_mask, annotations['mask'], transform_parameters)

                for index in range(annotations['poses'].shape[0]):
                    annotations['poses'][index, :] = adjust_pose_annotation(transform, annotations['poses'][index, :],
                                                                            annotations['cam_params'][index, :])
                    annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

            return image, annotations

        max_shape = (image_min_side, image_max_side, 3)

        seq = iaa.Sequential([
            #iaa.SomeOf((0, 2), [
            #    iaa.BlendAlphaFrequencyNoise(
            #        exponent=(-4, 4),
            #        foreground=iaa.MultiplyAndAddToBrightness((0.5, 1.5), (-50, 50)),
            #        background=iaa.MultiplyAndAddToBrightness((0.5, 1.5), (-50, 50)),
            #        upscale_method=["linear", "cubic"]
            #    ),
            #    iaa.BlendAlphaSimplexNoise(
            #        foreground=iaa.MultiplyAndAddToBrightness((0.5, 1.5), (-50, 50)),
            #        background=iaa.MultiplyAndAddToBrightness((0.5, 1.5), (-50, 50)),
            #        upscale_method=["linear", "cubic"]
            #    ),
            #    iaa.BlendAlphaSimplexNoise(
            #        foreground=[iaa.MultiplyHue((0.5, 1.5)), iaa.MultiplySaturation((0.5, 1.5))],
            #        background=[iaa.MultiplyHue((0.5, 1.5)), iaa.MultiplySaturation((0.5, 1.5))],
            #        upscale_method=["linear", "cubic"]
            #    ),
            #    iaa.BlendAlphaSimplexNoise(
            #        foreground=[iaa.AddToHue((-50, 50)), iaa.AddToSaturation((-50, 50))],
            #        background=[iaa.AddToHue((-50, 50)), iaa.AddToSaturation((-50, 50))],
            #        upscale_method=["linear", "cubic"]
            #    ),
            #]),
            # blur
            iaa.SomeOf((0, 2), [
                iaa.GaussianBlur((0.0, 2.0)),
                iaa.AverageBlur(k=(3, 7)),
                iaa.MedianBlur(k=(3, 7)),
                iaa.BilateralBlur(d=(1, 7)),
                iaa.MotionBlur(k=(3, 7))
            ]),
            # color
            iaa.SomeOf((0, 2), [
                # iaa.WithColorspace(),
                iaa.AddToHueAndSaturation((-15, 15)),
                # iaa.ChangeColorspace(to_colorspace[], alpha=0.5),
                iaa.Grayscale(alpha=(0.0, 0.2))
            ]),
            # brightness
            iaa.OneOf([
                iaa.Sequential([
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.Multiply((0.75, 1.25), per_channel=0.5)
                ]),
                iaa.Add((-10, 10), per_channel=0.5),
                iaa.Multiply((0.75, 1.25), per_channel=0.5),
                #iaa.BlendAlphaFrequencyNoise(
                #    exponent=(-4, 0),
                #    first=iaa.Multiply((0.75, 1.25), per_channel=0.5),
                #    second=iaa.LinearContrast((0.7, 1.3), per_channel=0.5))
                iaa.FrequencyNoiseAlpha(
                    exponent=(-4, 0),
                    first=iaa.Multiply((0.75, 1.25), per_channel=0.5),
                    second=iaa.LinearContrast((0.7, 1.3), per_channel=0.5))
            ]),
            # contrast
            iaa.SomeOf((0, 2), [
                iaa.GammaContrast((0.75, 1.25), per_channel=0.5),
                iaa.SigmoidContrast(gain=(0, 10), cutoff=(0.25, 0.75), per_channel=0.5),
                iaa.LogContrast(gain=(0.75, 1), per_channel=0.5),
                iaa.LinearContrast(alpha=(0.7, 1.3), per_channel=0.5)
            ]),
        ], random_order=True)

        while True:
            order = list(range(len(image_ids)))
            np.random.shuffle(order)
            groups = [[order[x % len(order)] for x in range(i, i + batch_size)] for i in
                          range(0, len(order), batch_size)]

            batches = np.arange(len(groups))

            for btx in range(len(batches)):
                x_s = [load_image(image_index) for image_index in groups[btx]]
                y_s = [load_annotations(image_index) for image_index in groups[btx]]

                assert (len(x_s) == len(y_s))

                # filter annotations
                for index, (image, annotations) in enumerate(zip(x_s, y_s)):

                    x_s[index] = augment_image(x_s[index], seq)

                    # transform a single group entry
                    x_s[index], y_s[index] = random_transform_group_entry(x_s[index], y_s[index])

                    # preprocess
                    x_s[index] = preprocess_image(x_s[index])
                    x_s[index] = keras.backend.cast_to_floatx(x_s[index])

                # x_s to image_batch
                image_source_batch = np.zeros((batch_size,) + max_shape, dtype=keras.backend.floatx())
                for image_index, image in enumerate(x_s):
                    image_source_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

                target_batch = compute_anchor_targets(x_s, y_s, len(classes))

                #image_source_batch = tf.convert_to_tensor(image_source_batch, dtype=tf.float32)
                #target_batch = tf.tuple(target_batch)

                #yield image_source_batch, target_batch
                #yield image_source_batch, target_batch
                yield image_source_batch, (target_batch[0], target_batch[1], target_batch[2], target_batch[3], target_batch[4])

    def __new__(self, data_dir, set_name, batch_size):

        if set_name=='val':
            return tf.data.Dataset.from_generator(self._sample,
                                              output_signature=(
                                                  tf.TensorSpec(shape=(1), dtype=tf.int64),
                                                  tf.TensorSpec(shape=(1), dtype=tf.int64),
                                                  tf.TensorSpec(shape=(480, 640, 3), dtype=tf.float32),
                                                  tf.TensorSpec(shape=(None, ), dtype=tf.float32),
                                                  tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                                  tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
                                                  tf.TensorSpec(shape=(None, 4), dtype=tf.float32)),
                                                  #{"labels": tf.TensorSpec(shape=(None,), dtype=tf.float64, name="labels")}),
                                                  # "bboxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float64, name="bboxes"),
                                                  # "poses": tf.TensorSpec(shape=(None, 7), dtype=tf.float64, name="poses"),
                                                  # "cam_params": tf.TensorSpec(shape=(None, 4), dtype=tf.float64, name="cam_params")}),
                                              args=(data_dir, set_name, batch_size))

        elif set_name == 'train':
            return tf.data.Dataset.from_generator(self._generate,
                                              output_signature=(tf.TensorSpec(shape=(batch_size, 480, 640, 3),dtype=tf.float32),
                                                                (tf.TensorSpec(shape=(batch_size, 6300, 15, 8, 17),dtype=tf.float32),
                                                                tf.TensorSpec(shape=(batch_size, 6300, 15 + 1),dtype=tf.float32),
                                                                tf.TensorSpec(shape=(batch_size, 6300, 15, 4),dtype=tf.float32),
                                                                tf.TensorSpec(shape=(batch_size, 6300, 15, 8, 7),dtype=tf.float32),
                                                                tf.TensorSpec(shape=(batch_size, 6300, 15), dtype=tf.float32))),
                                              args=(data_dir, set_name, batch_size))

        else:
            print('Define valid set_type for dataset generator [train, val].')

