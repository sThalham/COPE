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
from ..utils.image import read_image_bgr, read_image_dep
from collections import defaultdict

import os
import json
import numpy as np
import itertools
import yaml
import cv2


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class OcclusionGenerator(Generator):
    """ Generate data from the LineMOD dataset.
    """

    def __init__(self, data_dir, set_name, **kwargs):

        self.data_dir  = data_dir
        self.set_name  = set_name
        self.path      = os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json')
        self.mesh_info = os.path.join(data_dir, 'annotations', 'models_info' + '.yml')
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
            if "fx" in img:
                self.fx = img["fx"]
                self.fy = img["fy"]
                self.cx = img["cx"]
                self.cy = img["cy"]
            self.image_ids.append(img['id'])  # to correlate indexing to self.image_ann
        for ann in anno_ann:
            self.imgToAnns[ann['image_id']].append(ann)
            self.catToImgs[ann['category_id']].append(ann['image_id'])

        self.load_classes()

        self.TDboxes = np.ndarray((16, 8, 3), dtype=np.float32)

        for key, value in yaml.load(open(self.mesh_info)).items():
            x_minus = value['min_x']
            y_minus = value['min_y']
            z_minus = value['min_z']
            x_plus = value['size_x'] + x_minus
            y_plus = value['size_y'] + y_minus
            z_plus = value['size_z'] + z_minus
            three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                       [x_plus, y_plus, z_minus],
                                       [x_plus, y_minus, z_minus],
                                       [x_plus, y_minus, z_plus],
                                       [x_minus, y_plus, z_plus],
                                       [x_minus, y_plus, z_minus],
                                       [x_minus, y_minus, z_minus],
                                       [x_minus, y_minus, z_plus]])
            self.TDboxes[int(key), :, :] = three_box_solo

        super(OcclusionGenerator, self).__init__(**kwargs)

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
        path = path[:-4] + '_rgb' + path[-4:]

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

        return read_image_dep(path)

    def load_image_dep_raw(self, image_index):
        """ Load an image at the image_index.
        """
        if _isArrayLike(image_index):
            image_info = (self.image_ann[id] for id in image_index)
        elif type(image_index) == int:
            image_info = self.image_ann[image_index]
        path       = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
        # path = path[:-4] + '_dep.png'# + path[-4:]
        path = path[:-4] + '_dep_raw.png'

        return path

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
            CHECK DONE HERE: Annotations + images correct
        """
        # get ground truth annotations
        ids = self.image_ids[image_index]
        ids = ids if _isArrayLike(ids) else [ids]

        lists = [self.imgToAnns[imgId] for imgId in ids if imgId in self.imgToAnns]
        anns = list(itertools.chain.from_iterable(lists))

        # load mask
        if _isArrayLike(image_index):
            image_info = (self.image_ann[id] for id in image_index)
        elif type(image_index) == int:
            image_info = self.image_ann[image_index]
        path = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
        path = path[:-4] + '_mask.png'  # + path[-4:]
        # mask = None
        mask = cv2.imread(path, -1)

        annotations     = {'mask': mask, 'labels': np.empty((0,)), 'bboxes': np.empty((0, 4)), 'poses': np.empty((0, 7)), 'segmentations': np.empty((0, 8, 3)), 'cam_params': np.empty((0, 4)), 'mask_ids': np.empty((0,))}

        for idx, a in enumerate(anns):
            if self.set_name == 'train':
                if a['feature_visibility'] < 0.5:
                    continue

            annotations['labels'] = np.concatenate([annotations['labels'], [self.inv_label_to_label(a['category_id'])]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]]], axis=0)
            if a['pose'][2] < 10.0:  # needed for adjusting pose annotations
                a['pose'][0] = a['pose'][0] * 1000.0
                a['pose'][1] = a['pose'][1] * 1000.0
                a['pose'][2] = a['pose'][2] * 1000.0
            annotations['poses'] = np.concatenate([annotations['poses'], [[
                a['pose'][0],
                a['pose'][1],
                a['pose'][2],
                a['pose'][3],
                a['pose'][4],
                a['pose'][5],
                a['pose'][6],
            ]]], axis=0)
            annotations['mask_ids'] = np.concatenate([annotations['mask_ids'], [
                a['mask_id'],
            ]], axis=0)
            objID = a['category_id']
            #if objID > 5:
            #    objID = objID + 2
            #elif objID > 2:
            #    objID = objID + 1
            #else:
            #    objID = objID
            threeDbox = self.TDboxes[objID, :, :]
            annotations['segmentations'] = np.concatenate([annotations['segmentations'], [threeDbox]], axis=0)
            annotations['cam_params'] = np.concatenate([annotations['cam_params'], [[
                self.fx,
                self.fy,
                self.cx,
                self.cy,
            ]]], axis=0)

        return annotations
