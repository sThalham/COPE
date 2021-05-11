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

import numpy as np
import keras
import transforms3d as tf3d
import cv2
from PIL import Image

from ..utils.compute_overlap import compute_overlap


def anchor_targets_bbox(
    image_group,
    annotations_group,
    num_classes,
):
    """ Generate anchor targets for bbox detection.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        image_group: List of BGR images.
        annotations_group: List of annotations (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
                      where N is the number of anchors for an image and the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch that contains bounding-box regression targets for an image & anchor states (np.array of shape (batch_size, N, 4 + 1),
                      where N is the number of anchors for an image, the first 4 columns define regression targets for (x1, y1, x2, y2) and the
                      last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """

    assert(len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert(len(annotations_group) > 0), "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert('bboxes' in annotations), "Annotations should contain bboxes."
        assert('labels' in annotations), "Annotations should contain labels."
        assert('poses' in annotations), "Annotations should contain labels."
        assert('segmentations' in annotations), "Annotations should contain poses"

    batch_size = len(image_group)
    pyramid_levels = [3, 4, 5]
    image_shapes = guess_shapes(image_group[0].shape[:2], pyramid_levels)
    location_shape = int(image_shapes[0][1] * image_shapes[0][0]) + int(image_shapes[1][1] * image_shapes[1][0]) + int(image_shapes[2][1] * image_shapes[2][0])

    labels_batch      = np.zeros((batch_size, location_shape, num_classes + 1), dtype=keras.backend.floatx())
    regression_batch = np.zeros((batch_size, location_shape, 16 + 1), dtype=keras.backend.floatx())
    center_batch = np.zeros((batch_size, location_shape, 1 + 1), dtype=keras.backend.floatx())

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):

        # w/o mask
        mask = annotations['mask'][0]
        # w/o mask

        calculated_boxes = np.empty((0, 16))
        #anchors_spec = np.empty((0, ), dtype=np.int32)
        anchors_spec = []
        locations_spec = np.empty((0, 2))
        print('locations_spec shape init: ', locations_spec.shape)
        for idx, pose in enumerate(annotations['poses']):

            # mask part
            cls = int(annotations['labels'][idx])
            mask_id = annotations['mask_ids'][idx]
            for resx in range(len(image_shapes)):
                mask_flat = np.asarray(Image.fromarray(mask).resize((image_shapes[resx][1], image_shapes[resx][0]), Image.NEAREST))
                locations_level = np.where(mask_flat == int(mask_id))
                locations_level = np.concatenate([locations_level[0][:, np.newaxis], locations_level[1][:, np.newaxis]], axis=1)
                locations_spec = np.concatenate([locations_spec, locations_level], axis=0)

                mask_flat = mask_flat.flatten()
                anchors_pyramid = np.where(mask_flat == int(mask_id))
                anchors_spec.append(anchors_pyramid[0])

            anchors_spec = np.concatenate(anchors_spec, axis=0)
            if len(anchors_spec) > 1:
                labels_batch[index, anchors_spec, -1] = 1
                regression_batch[index, anchors_spec, -1] = 1
                center_batch[index, anchors_spec, -1] = 1

                #mask_batch[index, anchors_spec, cls] = 1

            rot = tf3d.quaternions.quat2mat(pose[3:])
            rot = np.asarray(rot, dtype=np.float32)
            tra = pose[:3]
            tDbox = rot[:3, :3].dot(annotations['segmentations'][idx].T).T
            tDbox = tDbox + np.repeat(tra[np.newaxis, 0:3], 8, axis=0)
            box3D = toPix_array(tDbox, fx=annotations['cam_params'][idx][0], fy=annotations['cam_params'][idx][1],
                                cx=annotations['cam_params'][idx][2], cy=annotations['cam_params'][idx][3])
            box3D = np.reshape(box3D, (16))
            calculated_boxes = np.concatenate([calculated_boxes, [box3D]], axis=0)

            regression[index, anchors_spec, :-1], centers[index, anchors_spec, :-1] = box3D_transform(box3D, locations_spec)

            '''
            # debug
            pose_aug = pose[7:]
            rot = tf3d.quaternions.quat2mat(pose_aug[3:])
            rot = np.asarray(rot, dtype=np.float32)
            tra = pose_aug[:3]
            tDbox = rot[:3, :3].dot(annotations['segmentations'][idx].T).T
            tDbox = tDbox + np.repeat(tra[np.newaxis, 0:3], 8, axis=0)
            box3D = toPix_array(tDbox, fx=annotations['cam_params'][idx][0], fy=annotations['cam_params'][idx][1],
                                cx=annotations['cam_params'][idx][2], cy=annotations['cam_params'][idx][3])
            box3D = np.reshape(box3D, (16))
            pose1 = box3D.reshape((16)).astype(np.int16)
            image_raw = image
            colEst = (0, 0, 255)
            image_raw = cv2.line(image_raw, tuple(pose1[0:2].ravel()), tuple(pose1[2:4].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose1[2:4].ravel()), tuple(pose1[4:6].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose1[4:6].ravel()), tuple(pose1[6:8].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose1[6:8].ravel()), tuple(pose1[0:2].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose1[0:2].ravel()), tuple(pose1[8:10].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose1[2:4].ravel()), tuple(pose1[10:12].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose1[4:6].ravel()), tuple(pose1[12:14].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose1[6:8].ravel()), tuple(pose1[14:16].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose1[8:10].ravel()), tuple(pose1[10:12].ravel()), colEst,
                                 5)
            image_raw = cv2.line(image_raw, tuple(pose1[10:12].ravel()), tuple(pose1[12:14].ravel()), colEst,
                                 5)
            image_raw = cv2.line(image_raw, tuple(pose1[12:14].ravel()), tuple(pose1[14:16].ravel()), colEst,
                                 5)
            image_raw = cv2.line(image_raw, tuple(pose1[14:16].ravel()), tuple(pose1[8:10].ravel()), colEst,
                                 5)
            pose = pose[:7]
            # debug
            '''

            '''
            pose = box3D.reshape((16)).astype(np.int16)
            image_raw = image
            colEst = (255, 0, 0)
            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 5)
            image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                                 5)
            image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                                 5)
            image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                                 5)
            image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                                 5)
                           
            cls_ind = np.where(annotations['labels']==cls) # index of cls
            if not len(cls_ind[0]) == 0:
                viz_img = True
                ov_laps = argmax_overlaps_inds #cls index overlap per anchor location?
                am_laps = positive_indices
                ov_laps = ov_laps[am_laps]
                pru_anc = anchors[am_laps, :]
                anc_idx = ov_laps == cls_ind
                true_anchors = pru_anc[anc_idx[0,:], :]
                for jdx in range(true_anchors.shape[0]):
                    bb = true_anchors[jdx, :]
                    #print(bb)
                    cv2.rectangle(image_raw, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                               (255, 255, 255), 2)
                    #cv2.rectangle(image_raw, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                    #              (255, 0, 0), 1)
                    #image_crop = image[0][int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2]), :]
                    #name = '/home/stefan/RGBDPose_viz/anno_' + str(rind) + '_' + str(cls) + '_' + str(jdx) + '_crop.jpg'
                    #cv2.imwrite(name, image_crop)
        if viz_img == True:
            #image_raw = image_raw[int(np.nanmin(true_anchors[:, 0])):int(np.nanmin(true_anchors[:, 1])), int(np.nanmax(true_anchors[:, 2])):int(np.nanmax(true_anchors[:, 3])), :]
            name = '/home/stefan/RGBDPose_viz/anno_' + str(rind) + '_RGB.jpg'
            cv2.imwrite(name, image_raw)
        '''
        #regression_3D[index, positive_indices, annotations['labels'][argmax_overlaps_inds[positive_indices]].astype(int), -1] = 1
        #rind = np.random.randint(0, 1000)
        #name = '/home/stefan/PyraPose_viz/anno_' + str(rind) + '_RGB.jpg'
        #cv2.imwrite(name, image_raw)
        #mask_viz = mask_viz.reshape((image_shapes[0][0], image_shapes[0][1], 3))
        #mask_viz = cv2.resize(mask_viz, (640, 480), interpolation=cv2.INTER_NEAREST)
        #name = '/home/stefan/PyraPose_viz/anno_' + str(rind) + '_MASK.jpg'
        #cv2.imwrite(name, mask_viz)

    return regression_batch, labels_batch, center_batch


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


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    anchor_params=None,
    shapes_callback=None,
):
    """ Generators anchors for a given shape.

    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.

    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        shifted_anchors = shift(image_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


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


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def box3D_transform(box, locations, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

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

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2, targets_dx3, targets_dy3, targets_dx4, targets_dy4, targets_dx5, targets_dy5, targets_dx6, targets_dy6, targets_dx7, targets_dy7, targets_dx8, targets_dy8))
    targets = targets.T

    targets = (targets - mean) / std
    #print(np.mean(gt_boxes, axis=0), np.var(gt_boxes, axis=0))

    ############################
    # here goes centerness calculation
    targets_x = np.stack(
        (targets_dx1, targets_dx2, targets_dx3, targets_dx4, targets_dx5, targets_dx6, targets_dx7, targets_dx8))
    targets_y = np.stack(
        (targets_dy1, targets_dy2, targets_dy3, targets_dy4, targets_dy5, targets_dy6, targets_dy7, targets_dy8))
    print('targets_x: ', targets_x.shape)

    return targets, centers


def toPix_array(translation, fx=None, fy=None, cx=None, cy=None):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy

    return np.stack((xpix, ypix), axis=1)

