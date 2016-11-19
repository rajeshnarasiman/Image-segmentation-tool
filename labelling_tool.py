# The MIT License (MIT)
#
# Copyright (c) 2015 University of East Anglia, Norwich, UK
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Developed by Geoffrey French in collaboration with Dr. M. Fisher and
# Dr. M. Mackiewicz.

import mimetypes, json, os, glob, copy, io, math

import numpy as np

import random

from PIL import Image, ImageDraw

from skimage import img_as_float
from skimage import transform
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage.util import pad
from skimage.measure import find_contours


class LabelClass (object):
    def __init__(self, name, human_name, colour):
        """
        Label class constructor
        
        :param name: identifier class name 
        :param human_name: human readable name
        :param colour: colour as a tuple or list e.g. [255, 0, 0] for red
        """
        self.name = name
        self.human_name = human_name
        colour = list(colour)
        if len(colour) != 3:
            raise TypeError, 'colour must be a tuple or list of length 3'
        self.colour = colour


    def to_json(self):
        return {'name': self.name, 'human_name': self.human_name, 'colour': self.colour}


def label_class(name, human_name, rgb):
    return {'name': name,
            'human_name': human_name,
            'colour': rgb}


def _next_wrapped_array(xs):
    return np.append(xs[1:], xs[:1], axis=0)

def _prev_wrapped_array(xs):
    return np.append(xs[-1:], xs[:-1], axis=0)

def _simplify_contour(cs):
    degenerate_verts = (cs == _next_wrapped_array(cs)).all(axis=1)
    while degenerate_verts.any():
        cs = cs[~degenerate_verts,:]
        degenerate_verts = (cs == _next_wrapped_array(cs)).all(axis=1)

    if cs.shape[0] > 0:
        # Degenerate eges
        edges = (_next_wrapped_array(cs) - cs)
        edges = edges / np.sqrt((edges**2).sum(axis=1))[:,None]
        degenerate_edges = (_prev_wrapped_array(edges) * edges).sum(axis=1) > (1.0 - 1.0e-6)
        cs = cs[~degenerate_edges,:]

        if cs.shape[0] > 0:
            return cs
    return None



class Vector2 (object):
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x + other.x, self.y + other.y)
        elif isinstance(other, float):
            return Vector2(self.x + other, self.y + other)
        else:
            raise TypeError('other must be a Vector2 or a float, not a {}'.format(type(other)))

    def __sub__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x - other.x, self.y - other.y)
        elif isinstance(other, float):
            return Vector2(self.x - other, self.y - other)
        else:
            raise TypeError('other must be a Vector2 or a float, not a {}'.format(type(other)))

    def __mul__(self, other):
        if isinstance(other, float):
            return Vector2(self.x + other.x, self.y + other.y)
        else:
            raise TypeError('other must be a float, not a {}'.format(type(other)))

    @staticmethod
    def min(a, b):
        return Vector2(min(a.x, b.x), min(a.y, b.y))

    @staticmethod
    def max(a, b):
        return Vector2(max(a.x, b.x), max(a.y, b.y))


    def as_json(self):
        return {'x': self.x, 'y': self.y}

    @classmethod
    def from_json(cls, j):
        return Vector2(j['x'], j['y'])

    @staticmethod
    def as_array(vertices):
        return np.array([[v.y, v.x] for v in vertices])

    @staticmethod
    def from_array(arr):
        return [Vector2(arr[i,1], arr[i,0]) for i in range(arr.shape[0])]


class AABox (object):
    def __init__(self, lower, upper):
        if not isinstance(lower, Vector2):
            raise TypeError('lower must be a Vector2')
        if not isinstance(upper, Vector2):
            raise TypeError('upper must be a Vector2')
        self.lower = lower
        self.upper = upper

    @property
    def centre(self):
        return (self.lower + self.upper) * 0.5

    @property
    def size(self):
        return self.upper - self.lower

    @staticmethod
    def from_points(points):
        if len(points) == 0:
            return None
        else:
            lower = upper = points[0]
            for p in points[1:]:
                lower = Vector2.min(lower, p)
                upper = Vector2.max(upper, p)
            return AABox(lower, upper)

    @staticmethod
    def from_boxes(boxes):
        if len(boxes) == 0:
            return None
        elif len(boxes) == 1:
            return boxes[0]
        else:
            lower = boxes[0].lower
            upper = boxes[0].upper
            for b in boxes[1:]:
                lower = Vector2.min(lower, b.lower)
                upper = Vector2.max(upper, b.upper)
            return AABox(lower, upper)


_LABEL_TYPE_REGISTRY = {}

def register_label_class(cls):
    _LABEL_TYPE_REGISTRY[cls.JSON_NAME] = cls
    return cls


class AbstractLabel (object):
    JSON_NAME = None

    def __init__(self, root, label_class=None, object_id=None):
        self.root = root
        self.label_class = label_class
        self.object_id = object_id
        self.aabox = None
        self.root.register_label(object_id, self)

    def warped(self, xform_fn):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))

    def render_mask(self, width, height, fill):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))

    def as_json(self):
        return {'label_type': self.JSON_NAME, 'label_class': self.label_class,
                'object_id': self.object_id}

    @classmethod
    def from_json(cls, root, j):
        raise NotImplementedError('Abstract for type {}'.format(cls))


@register_label_class
class PointLabel (AbstractLabel):
    JSON_NAME = 'point'

    def __init__(self, root, label_class=None, object_id=None, position=None):
        super(PointLabel, self).__init__(root, label_class, object_id)
        if position is None:
            position = Vector2()
        self.position = position
        self.aabox = AABox(position, position)

    def warped(self, xform_fn):
        """
        Warp the label given a warping function

        :param xform_fn: a transformation function of the form `f(vertices) -> warped_vertices`, where `vertices` and
        `warped_vertices` are both Numpy arrays of shape `(N,2)` where `N` is the number of vertices and the
        co-ordinates are `x,y` pairs. The transformations defined in `skimage.transform`, e.g. `AffineTransform` can
        be used here.
        :return: an AbstractLabel instance
        """
        warped_verts = Vector2.from_array(xform_fn(Vector2.as_array([self.position])))
        return PointLabel(warped_verts[0], self.label_class)

    def render_mask(self, width, height, fill):
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).point([(self.position.x, self.position.y)], fill=0)
        mask = np.array(img)
        return mask

    def as_json(self):
        j = super(PointLabel, self).as_json()
        j['position'] = self.position.as_json()
        return j

    @classmethod
    def from_json(cls, root, j):
        return PointLabel(root, label_class=j['label_class'], object_id=j['object_id'],
                          position=Vector2.from_json(j['position']))


@register_label_class
class BoxLabel (AbstractLabel):
    JSON_NAME = 'box'

    def __init__(self, root, label_class=None, object_id=None, centre=None, size=None):
        super(BoxLabel, self).__init__(root, label_class, object_id)
        if centre is None:
            centre = Vector2()
        if size is None:
            size = Vector2()
        self.centre = centre
        self.size = size
        self.aabox = AABox(centre - size * 0.5, centre + size * 0.5)

    def corners(self):
        half_size = self.size * 0.5
        return [
            self.centre + Vector2(-half_size.x, -half_size.y),
            self.centre + Vector2(half_size.x, -half_size.y),
            self.centre + Vector2(half_size.x, half_size.y),
            self.centre + Vector2(-half_size.x, half_size.y),
        ]

    def warped(self, xform_fn):
        """
        Warp the label given a warping function

        :param xform_fn: a transformation function of the form `f(vertices) -> warped_vertices`, where `vertices` and
        `warped_vertices` are both Numpy arrays of shape `(N,2)` where `N` is the number of vertices and the
        co-ordinates are `x,y` pairs. The transformations defined in `skimage.transform`, e.g. `AffineTransform` can
        be used here.
        :return: an AbstractLabel instance
        """
        warped_corners = Vector2.from_array(xform_fn(Vector2.as_array(self.corners())))
        warped_box = AABox.from_points(warped_corners)
        return BoxLabel(warped_box.centre, warped_box.size, self.label_class)

    def render_mask(self, width, height, fill):
        lower = self.aabox.lower
        upper = self.aabox.upper
        img = Image.new('L', (width, height), 0)
        if fill:
            ImageDraw.Draw(img).rectangle([(lower.x, lower.y), (upper.x, upper.y)], outline=1, fill=1)
        else:
            ImageDraw.Draw(img).rectangle([(lower.x, lower.y), (upper.x, upper.y)], outline=1, fill=0)
        mask = np.array(img)
        return mask

    def as_json(self):
        j = super(BoxLabel, self).as_json()
        j['centre'] = self.centre.as_json()
        j['size'] = self.size.as_json()
        return j

    @classmethod
    def from_json(cls, root, j):
        return BoxLabel(root, label_class=j['label_class'], object_id=j['object_id'],
                        centre=Vector2.from_json(j['centre']), size=Vector2.from_json(j['size']))


@register_label_class
class PolygonLabel (AbstractLabel):
    JSON_NAME = 'box'

    def __init__(self, root, label_class=None, object_id=None, vertices=None):
        super(PolygonLabel, self).__init__(root, label_class, object_id)
        if vertices is None:
            vertices = []
        self.vertices = vertices
        self.aabox = AABox.from_points(vertices)

    def warped(self, xform_fn):
        """
        Warp the label given a warping function

        :param xform_fn: a transformation function of the form `f(vertices) -> warped_vertices`, where `vertices` and
        `warped_vertices` are both Numpy arrays of shape `(N,2)` where `N` is the number of vertices and the
        co-ordinates are `x,y` pairs. The transformations defined in `skimage.transform`, e.g. `AffineTransform` can
        be used here.
        :return: an AbstractLabel instance
        """
        warped_verts = Vector2.from_array(xform_fn(Vector2.as_array(self.vertices)))
        return PolygonLabel(warped_verts, self.label_class)

    def render_mask(self, width, height, fill):
        polygon = [(v.x, v.y)  for v in self.vertices]
        img = Image.new('L', (width, height), 0)
        if fill:
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        else:
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=0)
        mask = np.array(img)
        return mask

    def as_json(self):
        j = super(PolygonLabel, self).as_json()
        j['vertices'] = [v.as_json() for v in self.vertices]
        return j

    @classmethod
    def from_json(cls, root, j):
        return PolygonLabel(root, label_class=j['label_class'], object_id=j['object_id'],
                            vertices=[Vector2.from_json(j_v) for j_v in j['vertices']])


@register_label_class
class CompositeLabel (AbstractLabel):
    JSON_NAME = 'composite'

    def __init__(self, root, label_class=None, object_id=None, component_object_ids=None):
        super(CompositeLabel, self).__init__(root, label_class, object_id)
        if component_object_ids is None:
            component_object_ids = []
        self.component_object_ids = component_object_ids
        self.__component_models = None
        self.aabox = None

    @property
    def component_models(self):
        if self.__component_models is None:
            self.__component_models = [self.root.get_label_by_id(obj_id) for obj_id in self.component_object_ids]
        return self.__component_models

    def warped(self, xform_fn):
        """
        Warp the label given a warping function

        :param xform_fn: a transformation function of the form `f(vertices) -> warped_vertices`, where `vertices` and
        `warped_vertices` are both Numpy arrays of shape `(N,2)` where `N` is the number of vertices and the
        co-ordinates are `x,y` pairs. The transformations defined in `skimage.transform`, e.g. `AffineTransform` can
        be used here.
        :return: an AbstractLabel instance
        """
        return CompositeLabel(self.component_object_ids, self.label_class, self.object_id)

    def render_mask(self, width, height, fill):
        return None

    def as_json(self):
        j = super(CompositeLabel, self).as_json()
        j['components'] = self.component_object_ids
        return j

    @classmethod
    def from_json(cls, root, j):
        return CompositeLabel(root=root, label_class=j['label_class'], object_id=j['object_id'],
                              component_object_ids=j['components'])


@register_label_class
class GroupLabel (AbstractLabel):
    JSON_NAME = 'group'

    def __init__(self, root, label_class=None, object_id=None, component_models=None):
        super(GroupLabel, self).__init__(root, label_class, object_id)
        if component_models is None:
            component_models = []
        self.component_models = component_models
        component_boxes = [model.aabox for model in component_models]
        self.aabox = AABox.from_boxes([box for box in component_boxes if box is not None])

    def warped(self, xform_fn):
        """
        Warp the label given a warping function

        :param xform_fn: a transformation function of the form `f(vertices) -> warped_vertices`, where `vertices` and
        `warped_vertices` are both Numpy arrays of shape `(N,2)` where `N` is the number of vertices and the
        co-ordinates are `x,y` pairs. The transformations defined in `skimage.transform`, e.g. `AffineTransform` can
        be used here.
        :return: an AbstractLabel instance
        """
        return GroupLabel([model.warped(xform_fn) for model in self.component_models],
                          self.label_class, self.object_id)

    def render_mask(self, width, height, fill):
        return None

    def as_json(self):
        j = super(GroupLabel, self).as_json()
        j['component_models'] = [model.as_json() for model in self.component_models]
        return j

    @classmethod
    def from_json(cls, root, j):
        return GroupLabel(root=root, label_class=j['label_class'], object_id=j['object_id'],
                          component_models=[json_to_label(j_l) for j_l in j['component_models']])



class ImageLabelsRoot (object):
    def __init__(self):
        self.labels = []
        self._object_id_to_label = {}


    def label_from_json(self, j):
        label_type_cls = _LABEL_TYPE_REGISTRY[j['label_type']]
        return label_type_cls.from_json(j)


    def register_label(self, object_id, label):
        self._object_id_to_label[object_id] = label


    @classmethod
    def from_json(cls, j):
        pass



class ImageLabels (object):
    """
    Represents labels in vector format, stored in JSON form. Has methods for
    manipulating and rendering them.

    """
    def __init__(self, labels_json):
        self.labels_json = labels_json


    def __len__(self):
        return len(self.labels_json)

    def __getitem__(self, item):
        return self.labels_json[item]


    def warp(self, xform_fn):
        """
        Warp the labels given a warping function

        :param xform_fn: a transformation function of the form `f(vertices) -> warped_vertices`, where `vertices` and
        `warped_vertices` are both Numpy arrays of shape `(N,2)` where `N` is the number of vertices and the
        co-ordinates are `x,y` pairs. The transformations defined in `skimage.transform`, e.g. `AffineTransform` can
        be used here.
        :return: an `ImageLabels` instance that contains the warped labels
        """
        labels = copy.deepcopy(self.labels_json)
        for label in labels:
            label_type = label['label_type']
            if label_type == 'polygon':
                # Polygonal label
                vertices = label['vertices']
                polygon = [[v['x'], v['y']]  for v in vertices]
                polygon = xform_fn(np.array(polygon))
                transformed_verts = [{'x': polygon[i,0], 'y': polygon[i,1]}
                                     for i in xrange(len(polygon))]
                label['vertices'] = transformed_verts
            elif label_type == 'composite':
                # Nothing to do
                pass
            else:
                raise TypeError, 'Unknown label type {0}'.format(label_type)
        return ImageLabels(labels)


    def render_labels(self, label_classes, image_shape, pixels_as_vectors=False, fill=True):
        """
        Render the labels to create a label image

        :param label_classes: a sequence of classes. If an item is a list or tuple, the classes contained
            within are mapped to the same label index.
            Each class should be a `LabelClass` instance, a string.
        :param image_shape: `(height, width)` tuple specifying the shape of the image to be returned
        :param pixels_as_vectors: If `False`, return an (height,width) array of dtype=int with pixels numbered
            according to their label. If `True`, return a (height,width,n_labels) array of dtype=float32 with each pixel
            being a feature vector that gives the weight of each label, where n_labels is `len(label_classes)`
        :param fill: if True, labels will be filled, otherwise they will be outlined
        :return: (H,W) array with dtype=int if pixels_as_vectors is False, otherwise (H,W,n_labels) with dtype=float32
        """
        if isinstance(label_classes, list) or isinstance(label_classes, tuple):
            cls_to_index = {}
            for i, cls in enumerate(label_classes):
                if isinstance(cls, LabelClass):
                    cls_to_index[cls.name] = i
                elif isinstance(cls, str)  or  isinstance(cls, unicode)  or  cls is None:
                    cls_to_index[cls] = i
                elif isinstance(cls, list)  or  isinstance(cls, tuple):
                    for c in cls:
                        if isinstance(c, LabelClass):
                            cls_to_index[c.name] = i
                        elif isinstance(c, str)  or  isinstance(c, unicode)  or  c is None:
                            cls_to_index[c] = i
                        else:
                            raise TypeError, 'Item {0} in label_classes is a list that contains an item that is not a LabelClass or a string but a {1}'.format(i, type(c).__name__)
                else:
                    raise TypeError, 'Item {0} in label_classes is not a LabelClass, string or list, but a {1}'.format(i, type(cls).__name__)
        else:
            raise TypeError, 'label_classes must be a sequence that can contain LabelClass instances, strings or sub-sequences of the former'


        height, width = image_shape

        if pixels_as_vectors:
            label_image = np.zeros((height, width, len(label_classes)), dtype='float32')
        else:
            label_image = np.zeros((height, width), dtype=int)

        for label in self.labels_json:
            label_type = label['label_type']
            label_cls_n = cls_to_index.get(label['label_class'], None)
            if label_cls_n is not None:
                if label_type == 'polygon':
                    # Polygonal label
                    vertices = label['vertices']
                    if len(vertices) >= 3:
                        polygon = [(v['x'], v['y'])  for v in vertices]

                        img = Image.new('L', (width, height), 0)
                        if fill:
                            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                        else:
                            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=0)
                        mask = np.array(img)

                        if pixels_as_vectors:
                            label_image[:,:,label_cls_n] += mask
                            label_image[:,:,label_cls_n] = np.clip(label_image[:,:,label_cls_n], 0.0, 1.0)
                        else:
                            label_image[mask >= 0.5] = label_cls_n + 1

                elif label_type == 'composite':
                    pass
                else:
                    raise TypeError, 'Unknown label type {0}'.format(label_type)

        return label_image


    def render_individual_labels(self, label_classes, image_shape, fill=True):
        """
        Render individual labels to create a label image.
        The resulting image is a multi-channel image, with a channel for each class in `label_classes`.
        Each individual label's class is used to select the channel that it is rendered into.
        Each label is given a different index that is rendered into the resulting image.

        :param label_classes: a sequence of classes. If an item is a list or tuple, the classes contained
            within are mapped to the same label index.
            Each class should be a `LabelClass` instance, a string.
            Each entry within label_classes will have a corresponding channel in the output image
        :param image_shape: `(height, width)` tuple specifying the shape of the image to be returned
        :param fill: if True, labels will be filled, otherwise they will be outlined
        :param image_shape: `None`, or a `(height, width)` tuple specifying the shape of the image to be rendered
        :return: tuple of (label_image, label_counts) where:
            label_image is a (H,W,C) array with dtype=int
            label_counts is a 1D array of length C (number of channels) that contains the number of labels drawn for each channel; effectively the maximum value found in each channel
        """
        # Create `cls_to_channel`
        if isinstance(label_classes, list) or isinstance(label_classes, tuple):
            cls_to_channel = {}
            for i, cls in enumerate(label_classes):
                if isinstance(cls, LabelClass):
                    cls_to_channel[cls.name] = i
                elif isinstance(cls, str)  or  isinstance(cls, unicode)  or  cls is None:
                    cls_to_channel[cls] = i
                elif isinstance(cls, list)  or  isinstance(cls, tuple):
                    for c in cls:
                        if isinstance(c, LabelClass):
                            cls_to_channel[c.name] = i
                        elif isinstance(c, str)  or  isinstance(c, unicode):
                            cls_to_channel[c] = i
                        else:
                            raise TypeError, 'Item {0} in label_classes is a list that contains an item that is not a LabelClass or a string but a {1}'.format(i, type(c).__name__)
                else:
                    raise TypeError, 'Item {0} in label_classes is not a LabelClass, string or list, but a {1}'.format(i, type(cls).__name__)
        else:
            raise TypeError, 'label_classes must be a sequence that can contain LabelClass instances, strings or sub-sequences of the former'


        height, width = image_shape

        label_image = np.zeros((height, width, len(label_classes)), dtype=int)

        channel_label_count = [0] * len(label_classes)

        for label in self.labels_json:
            label_type = label['label_type']
            label_channel = cls_to_channel.get(label['label_class'], None)
            if label_channel is not None:
                if label_type == 'polygon':
                    # Polygonal label
                    vertices = label['vertices']
                    if len(vertices) >= 3:
                        polygon = [(v['x'], v['y'])  for v in vertices]

                        img = Image.new('L', (width, height), 0)
                        if fill:
                            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                        else:
                            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=0)
                        mask = np.array(img)

                        value = channel_label_count[label_channel]
                        channel_label_count[label_channel] += 1

                        label_image[mask >= 0.5, label_channel] = value + 1
                elif label_type == 'composite':
                    pass
                else:
                    raise TypeError, 'Unknown label type {0}'.format(label_type)

        return label_image, np.array(channel_label_count)


    def extract_label_images(self, image_2d, label_class_set=None):
        """
        Extract an image of each labelled entity from a given image.
        The resulting image is the original image masked with an alpha channel that results from rendering the label

        :param image_2d: the image from which to extract images of labelled objects
        :param label_class_set: a sequence of classes whose labels should be rendered, or None for all labels
        :return: a list of (H,W,C) image arrays
        """
        image_shape = image_2d.shape[:2]

        label_images = []

        for label in self.labels_json:
            label_type = label['label_type']
            if label_class_set is None  or  label['label_class'] in label_class_set:
                if label_type == 'polygon':
                    # Polygonal label
                    vertices = label['vertices']
                    if len(vertices) >= 3:
                        polygon = [(v['x'], v['y'])  for v in vertices]
                        np_poly = np.array(polygon)

                        lx = int(math.floor(np.min(np_poly[:,0])))
                        ly = int(math.floor(np.min(np_poly[:,1])))
                        ux = int(math.ceil(np.max(np_poly[:,0])))
                        uy = int(math.ceil(np.max(np_poly[:,1])))

                        # Given that the images and labels may have been warped by a transformation,
                        # there is no guarantee that they lie within the bounds of the image
                        lx = max(min(lx, image_shape[1]), 0)
                        ux = max(min(ux, image_shape[1]), 0)
                        ly = max(min(ly, image_shape[0]), 0)
                        uy = max(min(uy, image_shape[0]), 0)

                        w = ux - lx
                        h = uy - ly

                        if w > 0  and  h > 0:
                            np_box_poly = np_poly - np.array([[lx, ly]])
                            box_poly = [(np_box_poly[i,0], np_box_poly[i,1])   for i in xrange(np_box_poly.shape[0])]

                            img = Image.new('L', (w, h), 0)
                            ImageDraw.Draw(img).polygon(box_poly, outline=1, fill=1)
                            mask = np.array(img)

                            if (mask > 0).any():
                                img_box = image_2d[ly:uy, lx:ux]
                                if len(img_box.shape) == 2:
                                    # Convert greyscale image to RGB:
                                    img_box = gray2rgb(img_box)
                                # Append the mask as an alpha channel
                                object_img = np.append(img_box, mask[:,:,None], axis=2)

                                label_images.append(object_img)
                elif label_type == 'composite':
                    pass
                else:
                    raise TypeError, 'Unknown label type {0}'.format(label_type)

        return label_images


    @classmethod
    def from_contours(cls, list_of_contours, label_classes=None):
        """
        Convert a list of contours to an `ImageLabels` instance.

        :param list_of_contours: list of contours, where each contour is an `(N,2)` numpy array.
                where `N` is the number of vertices, each of which is a `(y,x)` pair.
        :param label_classes: [optional] a list of the same length as `list_of_contours` that provides
                the label class of each contour
        :return: an `ImageLabels` instance containing the labels extracted from the contours
        """
        labels = []
        if not isinstance(label_classes, list):
            label_classes = [label_classes] * len(list_of_contours)
        for contour, lcls in zip(list_of_contours, label_classes):
            vertices = [{'x': contour[i][1], 'y': contour[i][0]}   for i in xrange(contour.shape[0])]
            label = {
                'label_type': 'polygon',
                'label_class': lcls,
                'vertices': vertices
            }
            labels.append(label)
        return cls(labels)


    @classmethod
    def from_label_image(cls, labels):
        """
        Convert a integer label mask image to an `ImageLabels` instance.

        :param labels: a `(h,w)` numpy array of dtype `int32` that gives an integer label for each
                pixel in the image. Label values start at 1; pixels with a value of 0 will not be
                included in the returned labels.
        :return: an `ImageLabels` instance containing the labels extracted from the label mask image
        """
        contours = []
        for i in xrange(1, labels.max()+1):
            lmask = labels == i

            if lmask.sum() > 0:
                mask_positions = np.argwhere(lmask)
                (ystart, xstart), (ystop, xstop) = mask_positions.min(0), mask_positions.max(0) + 1

                if ystop >= ystart+1 and xstop >= xstart+1:
                    mask_trim = lmask[ystart:ystop, xstart:xstop]
                    mask_trim = pad(mask_trim, [(1,1), (1,1)], mode='constant').astype(np.float32)
                    cs = find_contours(mask_trim, 0.5)
                    for contour in cs:
                        simp = _simplify_contour(contour + np.array((ystart, xstart)) - np.array([[1.0, 1.0]]))
                        if simp is not None:
                            contours.append(simp)
        return cls.from_contours(contours)



class AbsractLabelledImage (object):
    def __init__(self):
        pass


    @property
    def pixels(self):
        raise NotImplementedError

    @property
    def image_shape(self):
        return self.pixels.shape[:2]

    def data_and_mime_type_and_size(self):
        raise NotImplementedError


    @property
    def labels(self):
        raise NotImplementedError

    @labels.setter
    def labels(self, l):
        raise NotImplementedError

    def has_labels(self):
        raise NotImplementedError

    @property
    def labels_json(self):
        labels = self.labels
        return labels.labels_json if labels is not None else None

    @labels_json.setter
    def labels_json(self, l):
        self.labels = ImageLabels(l)


    @property
    def complete(self):
        raise NotImplementedError

    @complete.setter
    def complete(self, c):
        raise NotImplementedError


    def warped(self, projection, sz_px):
        warped_pixels = transform.warp(self.pixels, projection.inverse)[:int(sz_px[0]),:int(sz_px[1])].astype('float32')
        warped_labels = self.labels.warp(projection)
        return InMemoryLabelledImage(warped_pixels, warped_labels)


    def render_labels(self, label_classes, pixels_as_vectors=False, fill=True):
        """
        Render the labels to create a label image

        :param label_classes: a sequence of classes. If an item is a list or tuple, the classes contained
            within are mapped to the same label index.
            Each class should be a `LabelClass` instance, a string.
        :param pixels_as_vectors: If `False`, return an (height,width) array of dtype=int with pixels numbered
            according to their label. If `True`, return a (height,width,n_labels) array of dtype=float32 with each pixel
            being a feature vector that gives the weight of each label, where n_labels is `len(label_classes)`
        :param fill: if True, labels will be filled, otherwise they will be outlined
        :return: (H,W) array with dtype=int if pixels_as_vectors is False, otherwise (H,W,n_labels) with dtype=float32
        """
        return self.labels.render_labels(label_classes, self.image_shape,
                                         pixels_as_vectors=pixels_as_vectors, fill=fill)


    def render_individual_labels(self, label_classes, fill=True):
        """
        Render individual labels to create a label image.
        The resulting image is a multi-channel image, with a channel for each class in `label_classes`.
        Each individual label's class is used to select the channel that it is rendered into.
        Each label is given a different index that is rendered into the resulting image.

        :param label_classes: a sequence of classes. If an item is a list or tuple, the classes contained
            within are mapped to the same label index.
            Each class should be a `LabelClass` instance, a string.
            Each entry within label_classes will have a corresponding channel in the output image
        :param fill: if True, labels will be filled, otherwise they will be outlined
        :param image_shape: `None`, or a `(height, width)` tuple specifying the shape of the image to be rendered
        :return: tuple of (label_image, label_counts) where:
            label_image is a (H,W,C) array with dtype=int
            label_counts is a 1D array of length C (number of channels) that contains the number of labels drawn for each channel; effectively the maximum value found in each channel
        """
        return self.labels.render_individual_labels(label_classes, self.image_shape, fill=fill)


    def extract_label_images(self, label_class_set=None):
        """
        Extract an image of each labelled entity.
        The resulting image is the original image masked with an alpha channel that results from rendering the label

        :param label_class_set: a sequence of classes whose labels should be rendered, or None for all labels
        :return: a list of (H,W,C) image arrays
        """
        return self.labels.extract_label_images(self.pixels, label_class_set=label_class_set)



class InMemoryLabelledImage (AbsractLabelledImage):
    def __init__(self, pixels, labels=None, complete=False):
        super(InMemoryLabelledImage, self).__init__()
        if labels is None:
            labels = ImageLabels([])
        self.__pixels = pixels
        self.__labels = labels
        self.__complete = complete


    @property
    def pixels(self):
        return self.__pixels

    def data_and_mime_type_and_size(self):
        buf = io.BytesIO()
        imsave(buf, self.__pixels, format='png')
        return buf.getvalue(), 'image/png', int(self.__pixels.shape[1]), int(self.__pixels.shape[0])



    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, l):
        self.__labels = l

    def has_labels(self):
        return True


    @property
    def complete(self):
        return self.__complete

    @complete.setter
    def complete(self, c):
        self.__complete = c





class PersistentLabelledImage (AbsractLabelledImage):
    def __init__(self, path, labels_dir=None, readonly=False):
        super(PersistentLabelledImage, self).__init__()
        self.__labels_path = self.__compute_labels_path(path, labels_dir=labels_dir)
        self.__image_path = path
        self.__pixels = None

        self.__labels = None
        self.__complete = None
        self.__readonly = readonly



    @property
    def pixels(self):
        if self.__pixels is None:
            self.__pixels = img_as_float(imread(self.__image_path))
        return self.__pixels

    def data_and_mime_type_and_size(self):
        if os.path.exists(self.__image_path):
            with open(self.__image_path, 'rb') as img:
                shape = self.image_shape
                return img.read(), mimetypes.guess_type(self.__image_path)[0], int(shape[1]), int(shape[0])


    @property
    def image_path(self):
        return self.__image_path

    @property
    def image_filename(self):
        return os.path.basename(self.__image_path)

    @property
    def image_name(self):
        return os.path.splitext(self.image_filename)[0]



    @property
    def labels(self):
        labels, complete = self.__get_label_data()
        return labels

    @labels.setter
    def labels(self, l):
        self.__set_label_data(l, self.complete)


    @property
    def complete(self):
        labels, complete = self.__get_label_data()
        return complete

    @complete.setter
    def complete(self, c):
        self.__set_label_data(self.labels, c)


    def has_labels(self):
        return os.path.exists(self.__labels_path)


    def __get_label_data(self):
        if self.__labels is None  or  self.__complete is None:
            self.__labels = self.__complete = None
            if os.path.exists(self.__labels_path):
                with open(self.__labels_path, 'r') as f:
                    try:
                        wrapped = json.load(f)
                    except ValueError:
                        pass
                    else:
                        self.__labels, self.__complete = self.__unwrap_labels(self.image_path, wrapped)
        return self.__labels, self.__complete

    def __set_label_data(self, labels, complete):
        self.__labels = labels
        self.__complete = complete
        if not self.__readonly:
            if labels is None  or  (len(labels) == 0 and not complete):
                # No data; delete the file
                if os.path.exists(self.__labels_path):
                    os.remove(self.__labels_path)
            else:
                wrapped = self.__wrap_labels(self.image_path, labels, complete)
                with open(self.__labels_path, 'w') as f:
                    json.dump(wrapped, f, indent=3)



    @staticmethod
    def __wrap_labels(image_path, labels, complete):
        image_filename = os.path.split(image_path)[1]
        return {'image_filename': image_filename,
                'labels': labels.labels_json,
                'complete': complete}

    @staticmethod
    def __unwrap_labels(image_path, wrapped_labels):
        if isinstance(wrapped_labels, dict):
            return ImageLabels(wrapped_labels['labels']), wrapped_labels.get('complete', False)
        elif isinstance(wrapped_labels, list):
            return ImageLabels(wrapped_labels), False
        else:
            raise TypeError, 'Labels loaded from file must either be a dict or a list, not a {0}'.format(type(wrapped_labels))


    @staticmethod
    def __compute_labels_path(path, labels_dir=None):
        p = os.path.splitext(path)[0] + '__labels.json'
        if labels_dir is not None:
            p = os.path.join(labels_dir, os.path.split(p)[1])
        return p


    @classmethod
    def for_directory(cls, dir_path, image_filename_pattern='*.png', with_labels_only=False, labels_dir=None, readonly=False):
        image_paths = glob.glob(os.path.join(dir_path, image_filename_pattern))
        if with_labels_only:
            return [PersistentLabelledImage(img_path, labels_dir=labels_dir, readonly=readonly)   for img_path in image_paths   if os.path.exists(cls.__compute_labels_path(img_path))]
        else:
            return [PersistentLabelledImage(img_path, labels_dir=labels_dir, readonly=readonly)   for img_path in image_paths]



class LabelledImageFile (AbsractLabelledImage):
    def __init__(self, path, labels=None, complete=False, on_set_labels=None):
        super(LabelledImageFile, self).__init__()
        if labels is None:
            labels = ImageLabels([])
        self.__labels = labels
        self.__complete = complete
        self.__image_path = path
        self.__pixels = None
        self.__on_set_labels = on_set_labels



    @property
    def pixels(self):
        if self.__pixels is None:
            self.__pixels = img_as_float(imread(self.__image_path))
        return self.__pixels

    def data_and_mime_type_and_size(self):
        if os.path.exists(self.__image_path):
            with open(self.__image_path, 'rb') as img:
                shape = self.image_shape
                return img.read(), mimetypes.guess_type(self.__image_path)[0], int(shape[1]), int(shape[0])


    @property
    def image_path(self):
        return self.__image_path

    @property
    def image_filename(self):
        return os.path.basename(self.__image_path)

    @property
    def image_name(self):
        return os.path.splitext(self.image_filename)[0]



    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, l):
        self.__labels = l
        if self.__on_set_labels is not None:
            self.__on_set_labels(l)


    def has_labels(self):
        return True


    @property
    def complete(self):
        return self.__complete

    @complete.setter
    def complete(self, c):
        self.__complete = c





def shuffle_images_without_labels(labelled_images):
    with_labels = [img   for img in labelled_images   if img.has_labels()]
    without_labels = [img   for img in labelled_images   if not img.has_labels()]
    random.shuffle(without_labels)
    return with_labels + without_labels








