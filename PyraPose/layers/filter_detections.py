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


def filter_detections(
    boxes3D,
    classification,
    poses,
    confidence,
    num_classes,
    score_threshold       = 0.35,
    iou_threshold       = 0.8,
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

        x1 = tf.math.maximum(a[:, :, 0], b[:, :, 0])
        y1 = tf.math.maximum(a[:, :, 1], b[:, :, 1])
        x2 = tf.math.minimum(a[:, :, 2], b[:, :, 2])
        y2 = tf.math.minimum(a[:, :, 3], b[:, :, 3])

        wid = x2 - x1 + 1
        hei = y2 - y1 + 1
        inter = wid * hei

        aarea = (a[:, :, 2] - a[:, :, 0] + 1.0) * (a[:, :, 3] - a[:, :, 1] + 1.0)
        barea = (b[:, :, 2] - b[:, :, 0] + 1.0) * (b[:, :, 3] - b[:, :, 1] + 1.0)

        # intersection over union overlap
        ovlap = tf.math.divide_no_nan(inter, (aarea + barea - inter))
        ovlap = tf.where(tf.math.less_equal(wid, 0.0), 0.0, ovlap)
        ovlap = tf.where(tf.math.less_equal(hei, 0.0), 0.0, ovlap)
        # set invalid entries to 0 overlap
        indicator = tf.where(tf.math.greater(ovlap, iou_threshold), 1.0, 0.0)
        tf.print('indicator pre: ', tf.shape(indicator), tf.reduce_sum(indicator))
        all_ind = tf.where(indicator==1)
        uni, _ = tf.unique(all_ind[:, 1])
        ind_filter = tf.map_fn(lambda x: tf.argmax(tf.cast(tf.equal(all_ind[:, 1], x), tf.int64)), uni)

        #indicator = tf.zeros_like(indicator)
        filtered_indices = tf.gather(all_ind, ind_filter, axis=0)
        value_updates = tf.constant(1.0, shape=(tf.shape(filtered_indices)[0]))
        indicator = tf.scatter_nd(filtered_indices, value_updates, tf.cast(tf.shape(indicator), dtpype=tf.int64))

        #indicator[filtered_indices].assign(1.0)
        #indicator = tf.sparse_to_dense(filtered_indices, tf.zeros_like(indicator), tf.ones(tf.shape(filtered_indices)[0]))
        tf.print('indicator post: ', tf.shape(indicator), tf.reduce_sum(indicator))

        return indicator

    def _filter_detections(scores, labels, boxes3D, poses, confidence):
        # threshold based on score
        indices = tf.where(tf.math.greater(scores, score_threshold))
        labels = tf.gather_nd(labels, indices)
        indices = tf.stack([indices[:, 0], labels], axis=1)

        boxes3D = tf.gather(boxes3D, indices[:, 0], axis=0)
        poses = tf.gather(poses, indices[:, 0], axis=0)
        confidence = tf.gather(confidence, indices[:, 0], axis=0)
        
        x_min = tf.math.reduce_min(boxes3D[:, ::2], axis=1)
        y_min = tf.math.reduce_min(boxes3D[:, 1::2], axis=1)
        x_max = tf.math.reduce_min(boxes3D[:, ::2], axis=1)
        y_max = tf.math.reduce_min(boxes3D[:, 1::2], axis=1)

        boxes = tf.stack([x_min, y_min, x_max, y_max], axis=1)

        true_ovlaps = boxoverlap(boxes)

        #true_ovlaps = tf.where(true_ovlaps == 0.0, 10000.0, true_ovlaps)
        broadcast_confidence = true_ovlaps * confidence
        sort_args = tf.argsort(broadcast_confidence, axis=1, direction='ASCENDING')
        sort_conf = tf.sort(broadcast_confidence, axis=1, direction='ASCENDING')
        conf_mask = tf.where(tf.math.greater(sort_conf, 0.0), 1.0, 0.0)
        conf_mask = tf.tile(conf_mask[:, :, tf.newaxis], [1, 1, 12])

        print('poses_tiled: ', poses_tiled)
        sorted_poses = tf.gather(poses, indices=sort_args)
        filt_poses = conf_mask * sorted_poses

        # set indices > n_hyps to zero
        # sum poses and divide by actual n_hyp
        # indicator = tf.convert_to_tensor(np.array([[1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
        # conf = tf.convert_to_tensor(np.array([0.09, 0.12, 0.21, 0.18])
        # poses = tf.random.uniform(shape=[4, 12])

        filtered_poses = tf.math.reduce_mean(sorted_poses[:, :3, :], axis=1)
        ov_tiled = tf.tile(true_ovlaps[:, :, tf.newaxis], [1, 1, 12])
        poses = tf.where(tf.math.reduce_sum(ov_tiled, axis=1) < 3, -1 * tf.ones([tf.shape(poses)[0], 12]), filtered_poses)

        #poses_con = []
        #for ov in range(int(true_ovlaps.shape[1])):
        #    filt_confs = true_ovlaps[:, ov] * confidence
        #    sort_conf = tf.argsort(filt_confs, direction='ASCENDING')
        #    if tf.reduce_sum(true_ovlaps[:, ov]) < 3.0:
        #        poses_con.append(-1 * tf.ones([12]))
        #    else:
        #        poses_con.append(tf.math.reduce_mean(poses[sort_conf[:3], :], axis=0))

        #poses_con = tf.concat(poses_con, axis=0)

        #broadcast_confidence = true_ovlaps * confidence
        #broadcast_poses = tf
        #tf.print('confidence: ', confidence)
        #tf.print('true_ovlaps: ', broadcast_confidence[:, 0])

        return indices, poses

    in_shape = tf.shape(boxes3D)
    classification = tf.reshape(classification, [in_shape[0] * in_shape[1], num_classes])
    boxes3D = tf.reshape(boxes3D, [in_shape[0] * in_shape[1], num_classes, 16])
    poses = tf.reshape(poses, [in_shape[0] * in_shape[1], num_classes, 12])
    confidence = tf.reshape(confidence, [in_shape[0] * in_shape[1], num_classes])

    all_indices = []
    all_poses = []
    for c in range(int(classification.shape[1])):
        scores = classification[:, c]
        labels = c * backend.ones((keras.backend.shape(scores)[0],), dtype='int64')
        indices, filt_poses = _filter_detections(scores, labels, boxes3D[:, c, :], poses[:, c, :], confidence[:, c])
        all_indices.append(indices)
        all_poses.append(filt_poses)
        #all_indices.append(_filter_detections(scores, labels, boxes3D, poses, confidence))

    # concatenate indices to single tensor
    indices = tf.concat(all_indices, axis=0)
    poses = tf.concat(all_poses, axis=0)

    # select top k
    #scores              = backend.gather_nd(classification, indices)
    scores              = tf.gather_nd(classification, indices)
    labels              = indices[:, 1]
    #scores, top_indices = backend.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))
    scores, top_indices = tf.math.top_k(scores, k=tf.math.minimum(max_detections, tf.shape(scores)[0]))

    # filter input using the final set of indices
    indices = keras.backend.gather(indices[:, 0], top_indices)
    #boxes3D = keras.backend.gather(boxes3D, indices)
    #translation = keras.backend.gather(translation, indices)
    #translation = keras.backend.gather(translation, indices)
    poses = tf.gather(poses, indices)
    #confidence          = keras.backend.gather(confidence, indices)
    labels = tf.gather(labels, top_indices)
    #locations = keras.backend.gather(locations, indices)

    #filter_class = tf.stack([tf.range(tf.shape(indices)[0]), tf.cast(indices, tf.int32)], axis=-1)
    #confidence = tf.gather_nd(confidence, filter_class)
    #confidence = tf.math.reduce_sum(confidence, axis=1)

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
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
        score_threshold       = 0.5,
        max_detections        = 300,
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
                num_classes= self.num_classes,
                score_threshold       = self.score_threshold,
                max_detections        = self.max_detections,
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
            'score_threshold'       : self.score_threshold,
            'max_detections'        : self.max_detections,
            'parallel_iterations'   : 32,
        })

        return config
