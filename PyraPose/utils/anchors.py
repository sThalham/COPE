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
import math

import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import transforms3d as tf3d
import cv2
from PIL import Image
import math
import matplotlib.pyplot as plt
import copy
from .visualization import Visualizer


def anchor_targets_bbox(
    image_group,
    annotations_group,
    num_classes,
):

    assert(len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert(len(annotations_group) > 0), "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert('labels' in annotations), "Annotations should contain labels."
        assert('poses' in annotations), "Annotations should contain labels."
        assert('segmentations' in annotations), "Annotations should contain poses"

    batch_size = len(image_group)
    pyramid_levels = [3, 4, 5]
    image_shapes = guess_shapes(image_group[0].shape[:2], pyramid_levels)
    location_shape = int(image_shapes[0][1] * image_shapes[0][0]) + int(image_shapes[1][1] * image_shapes[1][0]) + int(image_shapes[2][1] * image_shapes[2][0])
    location_offset = [0, int(image_shapes[0][1] * image_shapes[0][0]), int(image_shapes[0][1] * image_shapes[0][0]) + int(image_shapes[1][1] * image_shapes[1][0])]

    labels_batch        = np.zeros((batch_size, location_shape, num_classes + 1), dtype=keras.backend.floatx())
    #regression_batch    = np.zeros((batch_size, location_shape, 16 + 1), dtype=keras.backend.floatx())
    regression_batch = np.zeros((batch_size, location_shape, num_classes, 16 + 1), dtype=keras.backend.floatx())
    center_batch        = np.zeros((batch_size, location_shape, 1 + 1), dtype=keras.backend.floatx())

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):

        #VISU = Visualizer(image)

        image_locations = locations_for_shape(image.shape)
        # w/o mask
        mask = annotations['mask'][0]
        # vanilla
        masks_level = []
        for jdx, resx in enumerate(image_shapes):
            mask_level = np.asarray(Image.fromarray(mask).resize((resx[1], resx[0]), Image.NEAREST))
            masks_level.append(mask_level.flatten())

        calculated_boxes = np.empty((0, 16))

        for idx, pose in enumerate(annotations['poses']):

            locations_positive = []
            #labels_positive = []
            #labels_values = []
            cls = int(annotations['labels'][idx])
            mask_id = annotations['mask_ids'][idx]
            obj_diameter = annotations['diameters'][idx]
            #labels_cls = np.where(mask == mask_id, 255, 0).astype(np.uint8)
            for jdx, resx in enumerate(image_shapes):
                locations_level = np.where(masks_level[jdx] == int(mask_id))[0] + location_offset[jdx]
                locations_positive.append(locations_level)

            locations_positive_obj = np.concatenate(locations_positive, axis=0)

            if locations_positive_obj.shape[0] > 1:
                labels_batch[index, locations_positive_obj, -1] = 1
                labels_batch[index, locations_positive_obj, cls] = 1
                #labels_batch[index, labels_positive_obj, -1] = 1
                #labels_batch[index, labels_positive_obj, cls] = labels_values_obj

                #regression_batch[index, locations_positive_obj, -1] = 1 # commented for now since we use highest 50% centerness
                regression_batch[index, locations_positive_obj, cls, -1] = 1
                center_batch[index, locations_positive_obj, -1] = 1

                #center_batch[index, :, -1] = 1

                rot = tf3d.quaternions.quat2mat(pose[3:])
                rot = np.asarray(rot, dtype=np.float32)
                tra = pose[:3]
                tDbox = rot[:3, :3].dot(annotations['segmentations'][idx].T).T
                tDbox = tDbox + np.repeat(tra[np.newaxis, 0:3], 8, axis=0)
                box3D = toPix_array(tDbox, fx=annotations['cam_params'][idx][0], fy=annotations['cam_params'][idx][1],
                                           cx=annotations['cam_params'][idx][2], cy=annotations['cam_params'][idx][3])
                box3D = np.reshape(box3D, (16))
                calculated_boxes = np.concatenate([calculated_boxes, [box3D]], axis=0)

                # project object diameter
                proj_diameter = (obj_diameter * annotations['cam_params'][idx][0]) / tra[2]

                # top 50% centerness
                #index_filt, boxes, centers =  box3D_transform(box3D, image_locations[locations_positive_obj, :], obj_diameter, proj_diameter)
                #regression_batch[index, locations_positive_obj[index_filt], :-1] = boxes
                #regression_batch[index, locations_positive_obj[index_filt], -1] = 1
                #center_batch[index, locations_positive_obj, :-1] = centers
                #center_batch[index, locations_positive_obj, -1] = 1

                # vanilla
                #regression_batch[index, locations_positive_obj, :-1], center_batch[index, locations_positive_obj, :-1] = box3D_transform(box3D, image_locations[locations_positive_obj, :], obj_diameter, proj_diameter) # regression_batch[index, anchors_spec, :-1], center_batch[index, anchors_spec, :-1] = box3D_transform(box3D, locations_spec)

                # per class anno
                regression_batch[index, locations_positive_obj, cls, :-1], center_batch[index, locations_positive_obj, :-1] = box3D_transform(box3D, image_locations[locations_positive_obj, :], obj_diameter, proj_diameter)

                #print('target: ', np.nanmax(regression_batch[index, locations_positive_obj, cls, :-1]), np.nanmin(regression_batch[index, locations_positive_obj, cls, :-1]))

                #VISU.give_data(box3D, center_batch[index, ...])

        #VISU.print_img()

    #return regression_batch, labels_batch, center_batch
    return tf.convert_to_tensor(regression_batch), tf.convert_to_tensor(labels_batch), tf.convert_to_tensor(center_batch),


def layer_shapes(image_shape, model):
    """Compute layer shapes given input image shape and the model.

    Args
        image_shape: The shape of the image.
        model: The model to use for computing how the image shape is transformed in the pyramid.

    Returns
        A dictionary mapping layer names to image shapes.
    """
    shape = {
        model.layers[0].name: (None,) + image_shape,
    }

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            inputs = [shape[lr.name] for lr in node.inbound_layers]
            if not inputs:
                continue
            shape[layer.name] = layer.compute_output_shape(inputs[0] if len(inputs) == 1 else inputs)

    return shape


def make_shapes_callback(model):
    """ Make a function for getting the shape of the pyramid levels.
    """
    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape["P{}".format(level)][1:3] for level in pyramid_levels]
        return image_shapes

    return get_shapes


def guess_shapes(image_shape, pyramid_levels):
    """Guess shapes based on pyramid levels.

    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def locations_for_shape(
    image_shape,
    pyramid_levels=None,
    shapes_callback=None,
    distributions=None,
):

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5]

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    if distributions is None:


    # compute anchors over all pyramid levels
    all_locations = np.zeros((0, 2))
    for idx, p in enumerate(image_shapes):
        ny, nx = p
        sy = image_shape[0] / ny
        sx = image_shape[1] / nx

        y = np.linspace(0, image_shape[0]-sy, num=ny) + sy/2
        x = np.linspace(0, image_shape[1]-sx, num=nx) + sx/2

        xv, yv = np.meshgrid(x, y)

        locations_level = np.concatenate([xv.flatten()[:, np.newaxis], yv.flatten()[:, np.newaxis]], axis=1)

        all_locations     = np.append(all_locations, locations_level, axis=0)

    return all_locations


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def box3D_transform(box, locations, obj_diameter, proj_diameter, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    #np.seterr(invalid='raise')

    if mean is None:
        mean = np.full(16, 0)  # np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    if std is None:
        std = np.full(16, 0.7)  #5200 # np.array([1.3e3, 1.3e3, 1.3e3, 1.3e3, 1.3e3, 1.3e3, 1.3e3, 1.3e3, 1.3e3, 1.3e3, 1.3e3, 1.3e3, 1.3e3, 1.3e3, 1.3e3, 1.3e3])
        #std = np.full(16, 0.85) # with max dimension
        #std = np.full(16, 1.5) # with min dimension

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std) 
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    #print(box.shape)
    #print(locations.shape)

    targets_dx1 = locations[:, 0] - box[0]
    targets_dy1 = locations[:, 1] - box[1]
    targets_dx2 = locations[:, 0] - box[2]
    targets_dy2 = locations[:, 1] - box[3]
    targets_dx3 = locations[:, 0] - box[4]
    targets_dy3 = locations[:, 1] - box[5]
    targets_dx4 = locations[:, 0] - box[6]
    targets_dy4 = locations[:, 1] - box[7]
    targets_dx5 = locations[:, 0] - box[8]
    targets_dy5 = locations[:, 1] - box[9]
    targets_dx6 = locations[:, 0] - box[10]
    targets_dy6 = locations[:, 1] - box[11]
    targets_dx7 = locations[:, 0] - box[12]
    targets_dy7 = locations[:, 1] - box[13]
    targets_dx8 = locations[:, 0] - box[14]
    targets_dy8 = locations[:, 1] - box[15]

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2, targets_dx3, targets_dy3, targets_dx4, targets_dy4, targets_dx5, targets_dy5, targets_dx6, targets_dy6, targets_dx7, targets_dy7, targets_dx8, targets_dy8), axis=1)

    # targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2, targets_dx3, targets_dy3, targets_dx4, targets_dy4, targets_dx5, targets_dy5, targets_dx6, targets_dy6, targets_dx7, targets_dy7, targets_dx8, targets_dy8))
    #targets = targets.T
    #if math.nan in targets:
    #    print("NaN detected")
    targets = (targets - mean) / (std * obj_diameter)
    #if math.nan in targets:
    #    print("NaN detected")
    #print(np.mean(gt_boxes, axis=0), np.var(gt_boxes, axis=0))

    #print('box: ', box)
    #print('diameter: ', diameter)

    x_sum = np.abs(np.sum(targets[:, ::2], axis=1))
    y_sum = np.abs(np.sum(targets[:, 1::2], axis=1))
    #centerness = (x_sum + y_sum) / (proj_diameter * 0.01)
    centerness = (np.power(x_sum, 2) + np.power(y_sum, 2)) / (proj_diameter * 0.01)
    #centerness = (x_sum + y_sum) / (proj_diameter * 0.015) # with max dimension
    #centerness = (x_sum + y_sum) / (proj_diameter * 0.03) # with min dimension
    #centerness = (x_sum + y_sum) / (proj_diameter * 0.02)
    centerness = np.exp(-centerness)
    #print('new sample: ')
    #print(np.sort(centerness))

    #print('in anchor hypotheses: ')
    #print(targets.shape)
    #print(len(np.where(centerness > 0.5)[0]))

    # top 50% centerness
    #med_cent = np.median(centerness)
    #indices_med = np.argwhere(centerness>med_cent)

    return targets, centerness[:, np.newaxis]
    #return indices_med, targets[indices_med, :], centerness[:, np.newaxis]


def toPix_array(translation, fx=None, fy=None, cx=None, cy=None):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy

    return np.stack((xpix, ypix), axis=1)