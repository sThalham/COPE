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

import tensorflow.keras as keras
import tensorflow as tf
from .. import backend
import numpy as np


def filter_detections(
    boxes3D,
    boxes,
    classification,
    poses,
    confidence,
    num_classes,
    score_threshold       = 0.5,
    iou_threshold         = 0.5,
    pose_hyps             = 10,
    max_detections        = 100,
):
    """ Filter detections using the boxes and classification values.

    Args
        boxes                 : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification        : Tensor of shape (num_boxes, num_classes) containing the classification scores.
        other                 : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
        nms                   : Flag to enable/disable non maximum suppression.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """

    def _filter_detections(indices, labels, boxes3D, boxes, poses, confidence):

        # tf vec_map
        #classification, labels, boxes3D, poses, confidence = args
        #indices = tf.where(tf.math.greater(classification, score_threshold))
        # tf vec_map

        # threshold based on score
        #indices = tf.where(tf.math.greater(scores, score_threshold))
        labels = tf.gather_nd(labels, indices)
        #scores = tf.gather_nd(scores, indices)
        indices = tf.stack([indices[:, 0], labels], axis=1)

        boxes3D = tf.gather(boxes3D, indices[:, 0], axis=0)
        boxes = tf.gather(boxes, indices[:, 0], axis=0)
        poses = tf.gather(poses, indices[:, 0], axis=0)
        confidence = tf.gather(confidence, indices[:, 0], axis=0)
        
        #x_min = tf.math.reduce_min(boxes3D[:, ::2], axis=1)
        #y_min = tf.math.reduce_min(boxes3D[:, 1::2], axis=1)
        #x_max = tf.math.reduce_max(boxes3D[:, ::2], axis=1)
        #y_max = tf.math.reduce_max(boxes3D[:, 1::2], axis=1)

        #boxes = tf.stack([x_min, y_min, x_max, y_max], axis=1)
        ########################################

        a = tf.tile(boxes[tf.newaxis, :, :], [tf.shape(boxes)[0], 1, 1])
        b = tf.transpose(a, perm=[1, 0, 2])

        x1 = tf.math.maximum(a[:, :, 0], b[:, :, 0])
        y1 = tf.math.maximum(a[:, :, 1], b[:, :, 1])
        x2 = tf.math.minimum(a[:, :, 2], b[:, :, 2])
        y2 = tf.math.minimum(a[:, :, 3], b[:, :, 3])

        wid = x2 - x1 + 1
        hei = y2 - y1 + 1
        inter = wid * hei

        aarea = (a[:, :, 2] - a[:, :, 0] + 1) * (a[:, :, 3] - a[:, :, 1] + 1)
        barea = (b[:, :, 2] - b[:, :, 0] + 1) * (b[:, :, 3] - b[:, :, 1] + 1)

        # intersection over union overlap
        ovlap = tf.math.divide_no_nan(inter, (aarea + barea - inter))
        ovlap = tf.where(tf.math.less_equal(wid, 0.0), 0.0, ovlap)
        ovlap = tf.where(tf.math.less_equal(hei, 0.0), 0.0, ovlap)


        # set invalid entries to 0 overlap
        indicator = tf.where(tf.math.greater(ovlap, iou_threshold), 1.0, 0.0)
        #indicator = tf.where(tf.math.greater(ovlap, iou_threshold), ovlap, 0.0)

        max_col = tf.math.argmax(indicator, axis=1)
        rep_rows = tf.range(0, tf.shape(indicator)[0])
        filtered_indices = tf.concat([tf.cast(max_col, dtype=tf.int32)[:, tf.newaxis], rep_rows[:, tf.newaxis]], axis=1)
        value_updates = tf.math.reduce_max(indicator, axis=1)
        indicator = tf.scatter_nd(filtered_indices, value_updates, tf.cast(tf.shape(indicator), dtype=tf.int32))
        true_ovlaps = tf.cast(indicator, dtype=tf.float32)
        #################################################
        #true_ovlaps = boxoverlap(boxes)

        #tf.print('ovlaps: ', ovlap)
        #tf.print('true_ovlaps: ', true_ovlaps)
        #tf.print('conf: ', confidence)

        ##########################################
        # projected
        #boxes_row = tf.tile(boxes3D[:, tf.newaxis, :], [1, tf.shape(boxes3D)[0], 1])
        #boxes_col = tf.tile(boxes3D[tf.newaxis, :, :], [tf.shape(boxes3D)[0], 1, 1])
        #box3D_ov = tf.math.abs(boxes_row - boxes_col)
        #box3D_ov = tf.math.reduce_sum(box3D_ov, axis=2)
        '''
        x_min = tf.math.reduce_min(boxes3D[:, ::2], axis=1)
        y_min = tf.math.reduce_min(boxes3D[:, 1::2], axis=1)
        x_max = tf.math.reduce_max(boxes3D[:, ::2], axis=1)
        y_max = tf.math.reduce_max(boxes3D[:, 1::2], axis=1)

        box_from_3D = tf.stack([x_min, y_min, x_max, y_max], axis=1)
        ########################################

        a = tf.tile(box_from_3D[tf.newaxis, :, :], [tf.shape(boxes)[0], 1, 1])
        b = tf.transpose(a, perm=[1, 0, 2])

        x1 = tf.math.maximum(a[:, :, 0], b[:, :, 0])
        y1 = tf.math.maximum(a[:, :, 1], b[:, :, 1])
        x2 = tf.math.minimum(a[:, :, 2], b[:, :, 2])
        y2 = tf.math.minimum(a[:, :, 3], b[:, :, 3])

        wid = x2 - x1 + 1
        hei = y2 - y1 + 1
        inter = wid * hei

        aarea = (a[:, :, 2] - a[:, :, 0] + 1) * (a[:, :, 3] - a[:, :, 1] + 1)
        barea = (b[:, :, 2] - b[:, :, 0] + 1) * (b[:, :, 3] - b[:, :, 1] + 1)

        # intersection over union overlap
        ovlap3D = tf.math.divide_no_nan(inter, (aarea + barea - inter))
        ovlap3D = tf.where(tf.math.less_equal(wid, 0.0), 0.0, ovlap3D)
        ovlap3D = tf.where(tf.math.less_equal(hei, 0.0), 0.0, ovlap3D)
        '''
        ###################### end ovlap

        # including confidence
        #broadcast_confidence = true_ovlaps * confidence
        # 2D box as heuristics
        broadcast_confidence = (1.0 - ovlap) * true_ovlaps * confidence
        # 3D box l1 as heuristics
        #broadcast_confidence = box3D_ov * true_ovlaps * confidence

        #broadcast_confidence = true_ovlaps
        broadcast_confidence = tf.where(broadcast_confidence == 0, 1000.0, broadcast_confidence)
        sort_args = tf.argsort(broadcast_confidence, axis=1, direction='ASCENDING')
        sort_conf = tf.sort(broadcast_confidence, axis=1, direction='ASCENDING')
        conf_mask = tf.where(tf.math.equal(sort_conf, 1000.0), 0.0, 1.0)
        repeats = tf.math.minimum(pose_hyps + 1, tf.shape(poses)[0])
        n_hyps = tf.tile(repeats[tf.newaxis], [tf.shape(poses)[0]])
        p_hyps = tf.sequence_mask(n_hyps, maxlen=tf.shape(poses)[0], dtype=tf.float32)
        conf_mask_pose = conf_mask * p_hyps

        box_hyps = tf.sequence_mask(n_hyps, maxlen=tf.shape(poses)[0], dtype=tf.float32)
        conf_mask_box = conf_mask * box_hyps

        pose_mask = tf.tile(conf_mask_pose[:, :, tf.newaxis], [1, 1, 12])
        sorted_poses = tf.gather(poses, indices=sort_args)
        filt_poses = pose_mask * sorted_poses

        box_mask = tf.tile(conf_mask_box[:, :, tf.newaxis], [1, 1, 4])
        sorted_boxes = tf.gather(boxes, indices=sort_args)
        filt_boxes = box_mask * sorted_boxes

        #w/o confidence
        #conf_mask = tf.tile(true_ovlaps[:, :, tf.newaxis], [1, 1, 12])
        #filt_poses = conf_mask * poses

        denom = tf.math.reduce_sum(pose_mask, axis=1)
        mean_poses = tf.math.reduce_sum(filt_poses, axis=1)
        poses = tf.math.divide_no_nan(mean_poses, denom)

        denom = tf.math.reduce_sum(box_mask, axis=1)
        mean_boxes = tf.math.reduce_sum(filt_boxes, axis=1)
        boxes = tf.math.divide_no_nan(mean_boxes, denom)

        zero_vector = tf.zeros(shape=(tf.shape(poses)[0]), dtype=tf.float32)
        #bool_mask = tf.not_equal(tf.math.reduce_max(denom, axis=1), zero_vector)
        #zero_vector = tf.ones(shape=(tf.shape(poses)[0]), dtype=tf.float32) * 1.0

        bool_mask = tf.math.greater(tf.math.reduce_max(denom, axis=1), zero_vector)
        poses = tf.boolean_mask(poses, bool_mask, axis=0)
        indices = tf.boolean_mask(indices, bool_mask, axis=0)
        indices = tf.cast(indices, dtype=tf.int32)
        boxes = tf.boolean_mask(boxes, bool_mask, axis=0)

        #ind_savior = tf.ones([100, 2], tf.int32) * -1
        #pos_savior = tf.ones([100, 12], tf.float32) * -1
        #indices = tf.concat([indices, ind_savior[:(100-tf.shape(indices)[0]), :]], axis=0)
        #poses = tf.concat([poses, pos_savior[:(100-tf.shape(poses)[0]), :]], axis=0)

        return indices, poses, boxes

    in_shape = tf.shape(boxes3D)
    classification = tf.reshape(classification, [in_shape[0] * in_shape[1], num_classes])
    boxes3D = tf.reshape(boxes3D, [in_shape[0] * in_shape[1], num_classes, 16])
    boxes = tf.reshape(boxes, [in_shape[0] * in_shape[1], num_classes, 4])
    poses = tf.reshape(poses, [in_shape[0] * in_shape[1], num_classes, 12])
    confidence = tf.reshape(confidence, [in_shape[0] * in_shape[1], num_classes])

    def dummy_fn():
        return tf.cast(tf.ones([1, 2]) * -1.0, dtype=tf.int32), tf.ones([1, 12]) * -1.0, tf.ones([1, 4]) * -1.0

    ##############################################
    # inefficient loop over classes
    # replace with vectorized_map
    all_indices = []
    all_poses = []
    all_boxes = []
    for c in range(int(classification.shape[1])):
        scores = classification[:, c]
        labels = c * backend.ones((keras.backend.shape(scores)[0],), dtype='int64')
        indices = tf.where(tf.math.greater(scores, score_threshold))
        indices, filt_poses, filt_boxes = tf.cond(tf.math.greater(tf.shape(indices)[0], 0), lambda: _filter_detections(indices, labels, boxes3D[:, c, :], boxes[:, c, :], poses[:, c, :], confidence[:, c]), lambda: dummy_fn())
        all_indices.append(indices)
        all_poses.append(filt_poses)
        all_boxes.append(filt_boxes)
    # concatenate indices to single tensor
    indices = tf.concat(all_indices, axis=0)
    poses = tf.concat(all_poses, axis=0)
    boxes = tf.concat(all_boxes, axis=0)
    ################################################
    # --------------------------------------------
    # tensorflow while_loop
    #def cond(i, filt1, filt2, d1, d2, d3, d4):
    #    return i < num_classes

    #def loop_body(c, indices_filt, poses_filt, classification, boxes3D, poses, confidence):
    #    scores = classification[:, c]
    #    labels = c * backend.ones((keras.backend.shape(scores)[0],), dtype='int64')
    #    indices = tf.where(tf.math.greater(scores, score_threshold))
    #    indices, filt_poses = tf.cond(tf.math.greater(tf.shape(indices)[0], 0),
    #                                  lambda: _filter_detections(indices, labels, boxes3D[:, c, :], poses[:, c, :],
    #                                                             confidence[:, c]), lambda: dummy_fn())
    #    indices_filt = tf.concat([indices_filt, indices], axis=0)
    #    poses_filt = tf.concat([poses_filt, filt_poses], axis=0)
    #    return c+1, indices_filt, poses_filt, classification, boxes3D, poses, confidence
    #iter = tf.constant(0, dtype=tf.int64)
    #indices_filt = tf.zeros([0, 2], dtype=tf.int32)
    #poses_filt = tf.zeros([0, 12])
    #iter, indices_filt, poses_filt, classification, boxes3D, poses, confidence = tf.while_loop(cond, loop_body, [iter, indices_filt, poses_filt, classification, boxes3D, poses, confidence], shape_invariants=[iter.get_shape(),
    #                                               tf.TensorShape([None, None]), tf.TensorShape([None, None]), classification.get_shape(), boxes3D.get_shape(), poses.get_shape(), confidence.get_shape()])
    #indices = indices_filt
    #poses = poses_filt
    ################################################################################
    # --------------------------------------------
    # vectorized_map over classes
    #classification = tf.transpose(classification, perm=[1, 0])
    #boxes3D = tf.transpose(boxes3D, perm=[1, 0, 2])
    #poses = tf.transpose(poses, perm=[1, 0, 2])
    #confidence = tf.transpose(confidence, perm=[1, 0])

    #labels = tf.range(0, num_classes, dtype='int64')
    #labels = tf.tile(labels[:, tf.newaxis], [1, tf.shape(classification)[0]])

    #indices, poses = tf.map_fn(_filter_detections, (classification, labels, boxes3D, poses, confidence), dtype=(tf.int32, tf.float32))
    #indices = tf.reshape(indices, [tf.shape(indices)[0] * tf.shape(indices)[1], tf.shape(indices)[2]])
    #poses = tf.reshape(poses, [tf.shape(poses)[0] * tf.shape(poses)[1], tf.shape(poses)[2]])
    ######################################################################################
    # select top k
    #scores              = backend.gather_nd(classification, indices)
    scores              = tf.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = tf.math.top_k(scores, k=tf.math.minimum(max_detections, tf.shape(scores)[0]))

    # filter input using the final set of indices
    indices = indices[:, 0]

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - tf.shape(scores)[0])
    poses    = tf.pad(poses, [[0, pad_size], [0, 0]], constant_values=-1)
    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores      = backend.pad(scores, [[0, pad_size]], constant_values=-1)
    labels      = backend.pad(labels, [[0, pad_size]], constant_values=-1)
    labels      = keras.backend.cast(labels, 'int32')
    indices     = backend.pad(indices, [[0, pad_size]], constant_values=-1)
    indices     = keras.backend.cast(indices, 'int32')

    # set shapes, since we know what they are
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    poses.set_shape([max_detections, 12])
    boxes.set_shape([max_detections, 4])
    indices.set_shape([max_detections])

    return [scores, labels, poses, indices, boxes]


class FilterDetections(keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
        self,
        num_classes=None,
        score_threshold=0.5,
        iou_threshold=0.5,
        pose_hyps=10,
        max_detections=100,
        **kwargs
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            nms                   : Flag to enable/disable NMS.
            class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold       : Threshold used to prefilter the boxes with.
            max_detections        : Maximum number of detections to keep.
            parallel_iterations   : Number of batch items to process in parallel.
        """
        self.num_classes = num_classes
        self.score_threshold       = score_threshold
        self.iou_threshold = iou_threshold
        self.pose_hyps = pose_hyps
        self.max_detections        = max_detections
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes3D = inputs[0]
        boxes = inputs[1]
        classification = inputs[2]
        poses = inputs[3]
        confidence = inputs[4]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes3D = inputs[0]
            boxes = inputs[1]
            classification = inputs[2]
            poses = inputs[3]
            confidence = inputs[4]

            return filter_detections(
                boxes3D,
                boxes,
                classification,
                poses,
                confidence,
                num_classes             = self.num_classes,
                iou_threshold           = self.iou_threshold,
                score_threshold         = self.score_threshold,
                pose_hyps               = self.pose_hyps,
                max_detections          = self.max_detections,
            )

        # call filter_detections on each batch
        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes3D, boxes, classification, poses, confidence],
            dtype=[tf.float32, tf.int32, tf.float32, tf.int32, tf.float32],
            parallel_iterations=32
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
            (input_shape[2][0], self.max_detections, 12),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections, 4),
        ] + [
            tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][5:])) for i in range(5, len(input_shape))
        ]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'num_classes'           : self.num_classes,
            'iou_threshold'         : self.iou_threshold,
            'pose_hyps'             : self.pose_hyps,
            'score_threshold'       : self.score_threshold,
            'max_detections'        : self.max_detections,
            'parallel_iterations'   : 32,
        })

        return config
