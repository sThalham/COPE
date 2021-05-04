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
import math

import keras
import tensorflow as tf

from matplotlib import pyplot

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import PyraPose.bin  # noqa: F401
    __package__ = "PyraPose.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.eval import evaluate
from ..utils.keras_version import check_keras_version


#def get_session():
#    """ Construct a modified tf session.
#    """
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    return tf.Session(config=config)


def create_generator(args):
    """ Create generators for evaluation.
    """
    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        validation_generator = CocoGenerator(
            args.coco_path,
            'val',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'linemod':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.linemod import LinemodGenerator

        validation_generator = LinemodGenerator(
            args.linemod_path,
            'val',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'occlusion':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.occlusion import OcclusionGenerator

        validation_generator = OcclusionGenerator(
            args.occlusion_path,
            'val',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'ycbv':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.ycbv import YCBvGenerator

        validation_generator = YCBvGenerator(
            args.ycbv_path,
            'val',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'tless':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.tless import TlessGenerator

        validation_generator = TlessGenerator(
            args.tless_path,
            'val',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'homebrewed':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.homebrewed import HomebrewedGenerator

        validation_generator = HomebrewedGenerator(
            args.homebrewed_path,
            'val',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )

    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    linemod_parser = subparsers.add_parser('linemod')
    linemod_parser.add_argument('linemod_path', help='Path to dataset directory (ie. /tmp/LineMOD).')

    occlusion_parser = subparsers.add_parser('occlusion')
    occlusion_parser.add_argument('occlusion_path', help='Path to dataset directory (ie. /tmp/Occlusion).')

    ycbv_parser = subparsers.add_parser('ycbv')
    ycbv_parser.add_argument('ycbv_path', help='Path to dataset directory (ie. /tmp/ycbv).')

    tless_parser = subparsers.add_parser('tless')
    tless_parser.add_argument('tless_path', help='Path to dataset directory (ie. /tmp/Tless).')

    tless_parser = subparsers.add_parser('homebrewed')
    tless_parser.add_argument('homebrewed_path', help='Path to dataset directory (ie. /tmp/Homebrewed).')

    parser.add_argument('model',              help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.5, type=float)
    parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=480)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=640)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generator
    generator = create_generator(args)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)

    #filters3 = model.layers[355].get_weights()[0]
    #f_min, f_max = filters3.min(), filters3.max()
    #filters3 = (filters3 - f_min) / (f_max - f_min)

    #n_filters, ix = 6, 1
    #for i in range(n_filters):
    #    f = filters3[:, :, :, i]
    #    for j in range(3):
    #        ax = pyplot.subplot(n_filters, 3, ix)
    #        ax.set_xticks([])
    #        ax.set_yticks([])
    #        pyplot.imshow(f[:, :, j], cmap='brg')
    #        ix += 1
    #pyplot.show()

    #filter4 = model.layers[355].get_weights()[0]
    #filter5 = model.layers[355].get_weights()[0]

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(model, anchor_params=anchor_params)



    # print model summary
    print(model.summary())

    # start evaluation

    if args.dataset_type == 'linemod':
        from ..utils.linemod_eval import evaluate_linemod
        evaluate_linemod(generator, model, args.score_threshold)

    elif args.dataset_type == 'occlusion':
        from ..utils.occlusion_eval import evaluate_occlusion
        evaluate_occlusion(generator, model, args.score_threshold)

    elif args.dataset_type == 'ycbv':
        from ..utils.ycbv_eval import evaluate_ycbv
        evaluate_ycbv(generator, model, args.score_threshold)

    elif args.dataset_type == 'tless':
        from ..utils.tless_eval import evaluate_tless
        evaluate_tless(generator, model, args.score_threshold)

    elif args.dataset_type == 'homebrewed':
        from ..utils.homebrewed_eval import evaluate_homebrewed
        evaluate_homebrewed(generator, model, args.score_threshold)

    else:
         print('unknown dataset: ', args.dataset_type)


if __name__ == '__main__':
    main()
