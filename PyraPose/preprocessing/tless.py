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

from ..preprocessing.generator import Generator
from ..utils.image import read_image_bgr
from collections import defaultdict

import os
import json
import numpy as np
import itertools
import cv2


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class TlessGenerator(Generator):
    """ Generate data from the LineMOD dataset.
    """

    def __init__(self, data_dir, set_name, **kwargs):
        """ Initialize a LineMOD data generator.
        Args
            data_dir: Path to where the LineMOD dataset is stored.
            set_name: Name of the set to parse.
        """
        self.data_dir  = data_dir
        self.set_name  = set_name
        self.path      = os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json')
        with open(self.path, 'r') as js:
            data = json.load(js)

        self.image_ann = data["images"]
        anno_ann = data["annotations"]
        cat_ann = data["categories"]
        self.cats = {}
        self.image_ids = []
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        for cat in cat_ann:
            self.cats[cat['id']] = cat
        for img in self.image_ann:
            self.image_ids.append(img['id'])  # to correlate indexing to self.image_ann
        for ann in anno_ann:
            self.imgToAnns[ann['image_id']].append(ann)
            self.catToImgs[ann['category_id']].append(ann['image_id'])

        self.load_classes()

        super(TlessGenerator, self).__init__(**kwargs)

    def load_classes(self):
        """ Loads the class to label mapping (and inverse) for COCO.
        """

        categories = self.cats
        if _isArrayLike(categories):
            categories = [categories[id] for id in categories]
        elif type(categories) == int:
            categories = [categories[categories]]
        categories.sort(key=lambda x: x['id'])

        self.classes        = {}
        self.labels         = {}
        self.labels_inverse = {}
        for c in categories:
            self.labels[len(self.classes)] = c['id']
            self.labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels_rev = {}
        for key, value in self.classes.items():
            self.labels_rev[value] = key

    def size(self):

        return len(self.image_ids)

    def num_classes(self):

        return len(self.classes)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels_rev

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def inv_label_to_label(self, label):
        """ Map COCO label to the label as used in the network.
        COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
        """
        return self.labels_inverse[label]

    def inv_label_to_name(self, label):
        """ Map COCO label to name.
        """
        return self.label_to_name(self.label_to_label(label))

    def label_to_inv_label(self, label):
        """ Map label as used by the network to labels as used by COCO.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):

        if _isArrayLike(image_index):
            image = (self.image_ann[id] for id in image_index)
        elif type(image_index) == int:
            image = self.image_ann[image_index]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        if _isArrayLike(image_index):
            image_info = (self.image_ann[id] for id in image_index)
        elif type(image_index) == int:
            image_info = self.image_ann[image_index]
        path       = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])

        return read_image_bgr(path)

    def load_image_dep(self, image_index):
        """ Load an image at the image_index.
        """
        if _isArrayLike(image_index):
            image_info = (self.image_ann[id] for id in image_index)
        elif type(image_index) == int:
            image_info = self.image_ann[image_index]
        path       = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
        path = path[:-4] + '_dep' + path[-4:]

        return read_image_bgr(path)

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
            CHECK DONE HERE: Annotations + images correct
        """
        # get ground truth annotations
        ids = self.image_ids[image_index]
        ids = ids if _isArrayLike(ids) else [ids]

        lists = [self.imgToAnns[imgId] for imgId in ids if imgId in self.imgToAnns]
        anns = list(itertools.chain.from_iterable(lists))

        annotations     = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4)), 'poses': np.empty((0, 6)), 'segmentations': np.empty((0, 16)), 'K': np.empty((0, 4))}

        for idx, a in enumerate(anns):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotations['labels'] = np.concatenate([annotations['labels'], [self.inv_label_to_label(a['category_id'])]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]]], axis=0)
            annotations['poses'] = np.concatenate([annotations['poses'], [[
                a['pose'][0],
                a['pose'][1],
                a['pose'][2],
                a['pose'][3],
                a['pose'][4],
                a['pose'][5],
            ]]], axis=0)
            annotations['segmentations'] = np.concatenate([annotations['segmentations'], [[
                a['segmentation'][0],
                a['segmentation'][1],
                a['segmentation'][2],
                a['segmentation'][3],
                a['segmentation'][4],
                a['segmentation'][5],
                a['segmentation'][6],
                a['segmentation'][7],
                a['segmentation'][8],
                a['segmentation'][9],
                a['segmentation'][10],
                a['segmentation'][11],
                a['segmentation'][12],
                a['segmentation'][13],
                a['segmentation'][14],
                a['segmentation'][15],
            ]]], axis=0)
            #annotations['K'] = np.concatenate([annotations['K'], [[
            #    a['calib'][0],
            #    a['calib'][1],
            #    a['calib'][2],
            #    a['calib'][3],
            #]]], axis=0)

        return annotations
