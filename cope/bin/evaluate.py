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
import json

import tensorflow.keras as keras
import tensorflow as tf

from matplotlib import pyplot

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import cope.bin  # noqa: F401
    __package__ = "cope.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models

def create_generator(args):
    """ Create generators for evaluation.
    """
    from ..preprocessing.data_generator import GeneratorDataset

    mesh_info = os.path.join(args.data_path, 'meshes', 'models_info' + '.json')
    num_classes = len(json.load(open(mesh_info)).items())
    dataset = GeneratorDataset(args.data_path, 'val', num_classes=num_classes, batch_size=1)
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
        correspondences[int(key) - 1, :, :] = three_box_solo
        # sphere_diameters[int(key) - 1] = value['diameter']
        sphere_diameters[int(key) - 1] = norm_pts

    return dataset, num_classes, correspondences, sphere_diameters


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')

    parser.add_argument('dataset', help='Path to dataset directory (ie. /tmp/your_converted_dataset).')
    parser.add_argument('--model',              help='Path to trained model.')
    parser.add_argument('--data-path', help='Path to dataset directory (ie. /tmp/your_converted_dataset).')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.75, type=float)
    parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=480)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=640)

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    generator, num_classes, correspondences, obj_diameters = create_generator(args)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model)

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(model, diameters=obj_diameters, classes=num_classes)

    # print model summary
    print(model.summary())

    from ..utils.data_eval import evaluate_data
    evaluate_data(generator, model, args.dataset, args.data_path, args.score_threshold)


if __name__ == '__main__':
    main()
