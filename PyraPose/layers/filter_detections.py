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
    classification,
    poses,
    confidence,
    num_classes,
    score_threshold       = 0.35,
    iou_threshold         = 0.7,
    pose_hyps             = 3,
    max_detections        = 300,
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

    def boxoverlap(boxes):
        a = tf.tile(boxes[tf.newaxis, :, :], [tf.shape(boxes)[0], 1, 1])
        b = tf.transpose(a, perm=[1, 0, 2])
        tf.print('a: ', a[0, 1, :])
        tf.print('b: ', b[0, 1, :])

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
        tf.print()
        ovlap = tf.where(tf.math.less_equal(wid, 0.0), 0.0, ovlap)
        ovlap = tf.where(tf.math.less_equal(hei, 0.0), 0.0, ovlap)

        # set invalid entries to 0 overlap
        indicator = tf.where(tf.math.greater(ovlap, iou_threshold), 1.0, 0.0)
        all_ind = tf.where(indicator==1)
        uni, _ = tf.unique(all_ind[:, 1])

        # slow filter
        #ind_filter = tf.map_fn(lambda x: tf.argmax(tf.cast(tf.equal(all_ind[:, 1], x), tf.int64)), uni)
        #filtered_indices = tf.gather(all_ind, ind_filter, axis=0)
        #value_updates = tf.tile(tf.convert_to_tensor(np.array([1.0])), [tf.shape(filtered_indices)[0]])
        #indicator = tf.scatter_nd(filtered_indices, value_updates, tf.cast(tf.shape(indicator), dtype=tf.int64))

        # faster filter?
        # top-notch tensorizing #1
        max_col = tf.math.argmax(indicator, axis=1)
        rep_rows = tf.range(0, tf.shape(indicator)[0])
        filtered_indices = tf.concat([tf.cast(max_col, dtype=tf.int32)[:, tf.newaxis], rep_rows[:, tf.newaxis]], axis=1)
        value_updates = tf.math.reduce_max(indicator, axis=1)
        indicator = tf.scatter_nd(filtered_indices, value_updates, tf.cast(tf.shape(indicator), dtype=tf.int64))

        indicator = tf.cast(indicator, dtype=tf.float32)

        tf.print('indicator post: ', tf.shape(indicator), tf.reduce_sum(indicator))

        return indicator

    def _filter_detections(indices, labels, boxes3D, poses, confidence):
        # threshold based on score
        #indices = tf.where(tf.math.greater(scores, score_threshold))
        labels = tf.gather_nd(labels, indices)
        indices = tf.stack([indices[:, 0], labels], axis=1)

        boxes3D = tf.gather(boxes3D, indices[:, 0], axis=0)
        poses = tf.gather(poses, indices[:, 0], axis=0)
        confidence = tf.gather(confidence, indices[:, 0], axis=0)
        
        x_min = tf.math.reduce_min(boxes3D[:, ::2], axis=1)
        y_min = tf.math.reduce_min(boxes3D[:, 1::2], axis=1)
        x_max = tf.math.reduce_max(boxes3D[:, ::2], axis=1)
        y_max = tf.math.reduce_max(boxes3D[:, 1::2], axis=1)

        boxes = tf.stack([x_min, y_min, x_max, y_max], axis=1)

        true_ovlaps = boxoverlap(boxes)

        broadcast_confidence = true_ovlaps * confidence
        broadcast_confidence = tf.where(broadcast_confidence == 0, 1000.0, broadcast_confidence)
        sort_args = tf.argsort(broadcast_confidence, axis=1, direction='ASCENDING')
        sort_conf = tf.sort(broadcast_confidence, axis=1, direction='ASCENDING')
        conf_mask = tf.where(tf.math.equal(sort_conf, 1000.0), 0.0, 1.0)
        repeats = tf.math.minimum(pose_hyps, tf.shape(poses)[0])
        n_hyps = tf.tile(repeats[tf.newaxis], [tf.shape(poses)[0]])
        n_hyps = tf.sequence_mask(n_hyps, maxlen=tf.shape(poses)[0], dtype=tf.float32)

        conf_mask = conf_mask * n_hyps
        conf_mask = tf.tile(conf_mask[:, :, tf.newaxis], [1, 1, 12])

        sorted_poses = tf.gather(poses, indices=sort_args)
        filt_poses = conf_mask * sorted_poses

        denom = tf.math.reduce_sum(conf_mask, axis=1)
        mean_poses = tf.math.reduce_sum(filt_poses, axis=1)
        poses = tf.math.divide_no_nan(mean_poses, denom)

        zero_vector = tf.zeros(shape=(tf.shape(poses)[0]), dtype=tf.float32)
        bool_mask = tf.not_equal(tf.math.reduce_max(denom, axis=1), zero_vector)
        poses = tf.boolean_mask(poses, bool_mask, axis=0)
        indices = tf.boolean_mask(indices, bool_mask, axis=0)


        #returns = tf.concat([poses, indices], axis=0)

        return indices, poses

    in_shape = tf.shape(boxes3D)
    classification = tf.reshape(classification, [in_shape[0] * in_shape[1], num_classes])
    boxes3D = tf.reshape(boxes3D, [in_shape[0] * in_shape[1], num_classes, 16])
    poses = tf.reshape(poses, [in_shape[0] * in_shape[1], num_classes, 12])
    confidence = tf.reshape(confidence, [in_shape[0] * in_shape[1], num_classes])

    def dummy_fn():
        return tf.cast(tf.ones([1, 2]) * -1.0, dtype=tf.int64), tf.ones([1, 12]) * -1.0


    # replace with vectorized_map
    all_indices = []
    all_poses = []
    for c in range(int(classification.shape[1])):
        scores = classification[:, c]
        labels = c * backend.ones((keras.backend.shape(scores)[0],), dtype='int64')

        indices = tf.where(tf.math.greater(scores, score_threshold))
        indices, filt_poses = tf.cond(tf.math.greater(tf.shape(indices)[0], 0), lambda: _filter_detections(indices, labels, boxes3D[:, c, :], poses[:, c, :], confidence[:, c]), lambda: dummy_fn())
        #indices, filt_poses = _filter_detections(indices, labels, boxes3D[:, c, :], poses[:, c, :], confidence[:, c])
        all_indices.append(indices)
        all_poses.append(filt_poses)
        #all_indices.append(_filter_detections(scores, labels, boxes3D, poses, confidence))

    # concatenate indices to single tensor
    indices = tf.concat(all_indices, axis=0)
    poses = tf.concat(all_poses, axis=0)

    #tf.print('all_ poses: ', tf.shape(all_poses))
    #tf.print('indices: ', tf.shape(indices))

    # select top k
    #scores              = backend.gather_nd(classification, indices)
    scores              = tf.gather_nd(classification, indices)
    labels              = indices[:, 1]
    #scores, top_indices = backend.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))
    scores, top_indices = tf.math.top_k(scores, k=tf.math.minimum(max_detections, tf.shape(scores)[0]))

    # filter input using the final set of indices
    indices = indices[:, 0]
    #indices = keras.backend.gather(indices[:, 0], top_indices)
    #poses = tf.gather(poses, indices)
    #labels = tf.gather(labels, top_indices)

    #filter_class = tf.stack([tf.range(tf.shape(indices)[0]), tf.cast(indices, tf.int32)], axis=-1)
    #confidence = tf.gather_nd(confidence, filter_class)
    #confidence = tf.math.reduce_sum(confidence, axis=1)

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - tf.shape(scores)[0])
    #boxes3D     = backend.pad(boxes3D, [[0, pad_size], [0, 0]], constant_values=-1)
    #translation = backend.pad(translation, [[0, pad_size], [0, 0]], constant_values=-1)
    poses    = backend.pad(poses, [[0, pad_size], [0, 0]], constant_values=-1)
    #translation = backend.pad(translation, [[0, pad_size], [0, 0], [0, 0]], constant_values=-1)
    #rotation = backend.pad(rotation, [[0, pad_size], [0, 0], [0, 0]], constant_values=-1)
    #confidence  = backend.pad(confidence, [[0, pad_size], [0, 0]], constant_values=-1)
    #confidence = backend.pad(confidence, [[0, pad_size]], constant_values=-1)
    #locations   = backend.pad(locations, [[0, pad_size], [0, 0]], constant_values=-1)
    scores      = backend.pad(scores, [[0, pad_size]], constant_values=-1)
    labels      = backend.pad(labels, [[0, pad_size]], constant_values=-1)
    labels      = keras.backend.cast(labels, 'int32')
    indices     = backend.pad(indices, [[0, pad_size]], constant_values=-1)
    indices     = keras.backend.cast(indices, 'int32')

    #print('poses: ', poses)
    #labels_idx = tf.repeat(tf.repeat(labels[:, tf.newaxis, tf.newaxis], repeats=7, axis=2), repeats=15, axis=1)
    #poses = tf.gather(poses, tf.repeat(labels[:, tf.newaxis, tf.newaxis], repeats=7, axis=2))
    #poses = tf.gather_nd(poses, labels_idx)

    #print('labels: ', labels)
    #print('poses: ', poses)

    # set shapes, since we know what they are
    #boxes3D.set_shape([max_detections, 16])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    poses.set_shape([max_detections, 12])
    #rotation.set_shape([max_detections, num_classes, 6])
    #confidence.set_shape([max_detections])
    indices.set_shape([max_detections])
    #tf.print('rotation reshaped: ', tf.shape(rotation))
    #tf.print('labels reshaped: ', tf.unique_with_counts(labels))

    #return [boxes3D, locations, scores, labels, translation, rotation]
    return [scores, labels, poses, indices]


class FilterDetections(keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
        self,
        num_classes=None,
        score_threshold=0.35,
        iou_threshold=0.8,
        pose_hyps=3,
        max_detections=300,
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
        classification = inputs[1]
        poses = inputs[2]
        confidence = inputs[3]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes3D = inputs[0]
            classification = inputs[1]
            poses = inputs[2]
            confidence = inputs[3]

            return filter_detections(
                boxes3D,
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
            elems=[boxes3D, classification, poses, confidence],
            dtype=[tf.float32, tf.int32, tf.float32, tf.int32],
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
        ] + [
            tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][4:])) for i in range(4, len(input_shape))
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
