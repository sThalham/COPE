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
    locations,
    poses,
    score_threshold       = 0.5,
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
    def _filter_detections(scores, labels):
        # threshold based on score
        indices = backend.where(keras.backend.greater(scores, score_threshold))

        '''
        nms = False
        if nms:
            filtered_boxes  = backend.gather_nd(boxes, indices)
            filtered_scores = keras.backend.gather(scores, indices)[:, 0]

            # perform NMS
            nms_indices = backend.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)

            # filter indices based on NMS
            indices = keras.backend.gather(indices, nms_indices)
        '''

        # add indices to list of all indices
        labels = backend.gather_nd(labels, indices)
        indices = keras.backend.stack([indices[:, 0], labels], axis=1)

        return indices

    all_indices = []
    for c in range(int(classification.shape[1])):
        scores = classification[:, c]
        labels = c * backend.ones((keras.backend.shape(scores)[0],), dtype='int64')
        all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
    indices = keras.backend.concatenate(all_indices, axis=0)

    # select top k
    scores              = backend.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = backend.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices             = keras.backend.gather(indices[:, 0], top_indices)
    boxes3D             = keras.backend.gather(boxes3D, indices)
    poses             = keras.backend.gather(poses, indices)
    labels              = keras.backend.gather(labels, top_indices)
    locations          = keras.backend.gather(locations, indices)

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes3D     = backend.pad(boxes3D, [[0, pad_size], [0, 0]], constant_values=-1)
    poses       = backend.pad(poses, [[0, pad_size], [0, 0], [0, 0]], constant_values=-1)
    locations   = backend.pad(locations, [[0, pad_size], [0, 0]], constant_values=-1)
    scores      = backend.pad(scores, [[0, pad_size]], constant_values=-1)
    labels      = backend.pad(labels, [[0, pad_size]], constant_values=-1)
    labels      = keras.backend.cast(labels, 'int32')

    #print('poses: ', poses)
    #labels_idx = tf.repeat(tf.repeat(labels[:, tf.newaxis, tf.newaxis], repeats=7, axis=2), repeats=15, axis=1)
    #poses = tf.gather(poses, tf.repeat(labels[:, tf.newaxis, tf.newaxis], repeats=7, axis=2))
    #poses = tf.gather_nd(poses, labels_idx)

    #print('labels: ', labels)
    #print('poses: ', poses)

    # set shapes, since we know what they are
    boxes3D.set_shape([max_detections, 16])
    locations.set_shape([max_detections, 2])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    poses.set_shape([max_detections, 15, 7])

    return [boxes3D, locations, scores, labels, poses]


class FilterDetections(keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
        self,
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
        locations = inputs[2]
        poses = inputs[3]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes3D = args[0]
            classification = args[1]
            locations = args[2]
            poses = args[3]

            return filter_detections(
                boxes3D,
                classification,
                locations,
                poses,
                score_threshold       = self.score_threshold,
                max_detections        = self.max_detections,
            )

        # call filter_detections on each batch
        outputs = backend.map_fn(
            _filter_detections,
            elems=[boxes3D, classification, locations, poses],
            dtype=[keras.backend.floatx(), keras.backend.floatx(), keras.backend.floatx(), 'int32', keras.backend.floatx()],
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
            (input_shape[0][0], self.max_detections, 16),
            (input_shape[2][0], self.max_detections, 2),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
            (input_shape[3][0], self.max_detections, 15, 7),
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
            'score_threshold'       : self.score_threshold,
            'max_detections'        : self.max_detections,
            'parallel_iterations'   : 32,
        })

        return config
