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

import cv2
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from PIL import Image

from .colors import label_color


class Visualizer:

    def __init__(self, image):
        self.image_raw = image
        self.image_raw[..., 0] += 103.939
        self.image_raw[..., 1] += 116.779
        self.image_raw[..., 2] += 123.68
        self.image_cen = copy.deepcopy(self.image_raw)

    def give_data(self, box3D, centers):

        pose = box3D.reshape((16)).astype(np.int16)
        colEst = (255, 0, 0)
        self.image_raw = cv2.line(self.image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 5)
        self.image_raw = cv2.line(self.image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 5)
        self.image_raw = cv2.line(self.image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 5)
        self.image_raw = cv2.line(self.image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 5)
        self.image_raw = cv2.line(self.image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 5)
        self.image_raw = cv2.line(self.image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 5)
        self.image_raw = cv2.line(self.image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 5)
        self.image_raw = cv2.line(self.image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 5)
        self.image_raw = cv2.line(self.image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                             5)
        self.image_raw = cv2.line(self.image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                             5)
        self.image_raw = cv2.line(self.image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                             5)
        self.image_raw = cv2.line(self.image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                             5)

        # image_center = center_batch[index, :4800, :-1].reshape((60,80))
        # image_center_g = np.where(image_center > 0.5, 255, 0).astype(np.uint8)
        # img_temp = np.where(image_center > 0.5, 0, image_center)
        # image_center_b = np.where(img_temp > 0, 255, 0).astype(np.uint8)
        # image_center_r = np.zeros((60,80), dtype=np.uint8)
        # image_center = np.concatenate([image_center_b[:, :, np.newaxis], image_center_g[:, :, np.newaxis], image_center_r[:, :, np.newaxis]], axis=2)

        image_center = centers[:4800, :-1].reshape((60, 80))
        image_center = (image_center*255).astype(np.uint8)
        image_center = np.repeat(image_center[:, :, np.newaxis], repeats=3, axis=2)
        self.image_center = np.asarray(Image.fromarray(image_center).resize((640, 480), Image.NEAREST))

    def print_img(self, path=None):
        image_viz = np.concatenate([self.image_raw.astype(np.uint8), self.image_center.astype(np.uint8)], axis=1)
        if path == None:

            plt.imshow(image_viz)
            plt.show()
        else:
            rind = np.random.randint(0, 1000)
            name = os.path.join(path, 'anno_' + str(rind) + '_RGB.jpg')
            cv2.imwrite(name, image_viz)


def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, color=(0, 255, 0), label_to_name=None):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary containing bboxes (shaped [N, 4]) and labels (shaped [N]).
        color         : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert('bboxes' in annotations)
    assert('labels' in annotations)
    assert(annotations['bboxes'].shape[0] == annotations['labels'].shape[0])

    for i in range(annotations['bboxes'].shape[0]):
        label   = annotations['labels'][i]
        c       = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, annotations['bboxes'][i], caption)
        draw_box(image, annotations['bboxes'][i], color=c)
