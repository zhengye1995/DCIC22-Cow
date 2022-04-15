# Copyright (c) OpenMMLab. All rights reserved.
import copy

import cv2
cv2.setNumThreads(0)
import mmcv
import numpy as np

from ..builder import PIPELINES
from .compose import Compose

_MAX_LEVEL = 10


def level_to_value(level, max_value):
    """Map from level to values based on max_value."""
    return (level / _MAX_LEVEL) * max_value


def enhance_level_to_value(level, a=1.8, b=0.1):
    """Map from level to values."""
    return (level / _MAX_LEVEL) * a + b


def random_negative(value, random_negative_prob):
    """Randomly negate value based on random_negative_prob."""
    return -value if np.random.rand() < random_negative_prob else value


def bbox2fields():
    """The key correspondence from bboxes to labels, masks and
    segmentations."""
    bbox2label = {
        'gt_bboxes': 'gt_labels',
        'gt_bboxes_ignore': 'gt_labels_ignore'
    }
    bbox2mask = {
        'gt_bboxes': 'gt_masks',
        'gt_bboxes_ignore': 'gt_masks_ignore'
    }
    bbox2seg = {
        'gt_bboxes': 'gt_semantic_seg',
    }
    return bbox2label, bbox2mask, bbox2seg


@PIPELINES.register_module()
class AutoAugment:
    """Auto augmentation.

    This data augmentation is proposed in `Learning Data Augmentation
    Strategies for Object Detection <https://arxiv.org/pdf/1906.11172>`_.

    TODO: Implement 'Shear', 'Sharpness' and 'Rotate' transforms

    Args:
        policies (list[list[dict]]): The policies of auto augmentation. Each
            policy in ``policies`` is a specific augmentation policy, and is
            composed by several augmentations (dict). When AutoAugment is
            called, a random policy in ``policies`` will be selected to
            augment images.

    Examples:
        >>> replace = (104, 116, 124)
        >>> policies = [
        >>>     [
        >>>         dict(type='Sharpness', prob=0.0, level=8),
        >>>         dict(
        >>>             type='Shear',
        >>>             prob=0.4,
        >>>             level=0,
        >>>             replace=replace,
        >>>             axis='x')
        >>>     ],
        >>>     [
        >>>         dict(
        >>>             type='Rotate',
        >>>             prob=0.6,
        >>>             level=10,
        >>>             replace=replace),
        >>>         dict(type='Color', prob=1.0, level=6)
        >>>     ]
        >>> ]
        >>> augmentation = AutoAugment(policies)
        >>> img = np.ones(100, 100, 3)
        >>> gt_bboxes = np.ones(10, 4)
        >>> results = dict(img=img, gt_bboxes=gt_bboxes)
        >>> results = augmentation(results)
    """

    def __init__(self, policies):
        assert isinstance(policies, list) and len(policies) > 0, \
            'Policies must be a non-empty list.'
        for policy in policies:
            assert isinstance(policy, list) and len(policy) > 0, \
                'Each policy in policies must be a non-empty list.'
            for augment in policy:
                assert isinstance(augment, dict) and 'type' in augment, \
                    'Each specific augmentation must be a dict with key' \
                    ' "type".'

        self.policies = copy.deepcopy(policies)
        self.transforms = [Compose(policy) for policy in self.policies]

    def __call__(self, results):
        transform = np.random.choice(self.transforms)
        return transform(results)

    def __repr__(self):
        return f'{self.__class__.__name__}(policies={self.policies})'


@PIPELINES.register_module()
class Shear:
    """Apply Shear Transformation to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        level (int | float): The level should be in range [0,_MAX_LEVEL].
        img_fill_val (int | float | tuple): The filled values for image border.
            If float, the same fill value will be used for all the three
            channels of image. If tuple, the should be 3 elements.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        prob (float): The probability for performing Shear and should be in
            range [0, 1].
        direction (str): The direction for shear, either "horizontal"
            or "vertical".
        max_shear_magnitude (float): The maximum magnitude for Shear
            transformation.
        random_negative_prob (float): The probability that turns the
                offset negative. Should be in range [0,1]
        interpolation (str): Same as in :func:`mmcv.imshear`.
    """

    def __init__(self,
                 level,
                 img_fill_val=128,
                 seg_ignore_label=255,
                 prob=0.5,
                 direction='horizontal',
                 max_shear_magnitude=0.3,
                 random_negative_prob=0.5,
                 interpolation='bilinear'):
        assert isinstance(level, (int, float)), 'The level must be type ' \
            f'int or float, got {type(level)}.'
        assert 0 <= level <= _MAX_LEVEL, 'The level should be in range ' \
            f'[0,{_MAX_LEVEL}], got {level}.'
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, 'img_fill_val as tuple must ' \
                f'have 3 elements. got {len(img_fill_val)}.'
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError(
                'img_fill_val must be float or tuple with 3 elements.')
        assert np.all([0 <= val <= 255 for val in img_fill_val]), 'all ' \
            'elements of img_fill_val should between range [0,255].' \
            f'got {img_fill_val}.'
        assert 0 <= prob <= 1.0, 'The probability of shear should be in ' \
            f'range [0,1]. got {prob}.'
        assert direction in ('horizontal', 'vertical'), 'direction must ' \
            f'in be either "horizontal" or "vertical". got {direction}.'
        assert isinstance(max_shear_magnitude, float), 'max_shear_magnitude ' \
            f'should be type float. got {type(max_shear_magnitude)}.'
        assert 0. <= max_shear_magnitude <= 1., 'Defaultly ' \
            'max_shear_magnitude should be in range [0,1]. ' \
            f'got {max_shear_magnitude}.'
        self.level = level
        self.magnitude = level_to_value(level, max_shear_magnitude)
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob
        self.direction = direction
        self.max_shear_magnitude = max_shear_magnitude
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation

    def _shear_img(self,
                   results,
                   magnitude,
                   direction='horizontal',
                   interpolation='bilinear'):
        """Shear the image.

        Args:
            results (dict): Result dict from loading pipeline.
            magnitude (int | float): The magnitude used for shear.
            direction (str): The direction for shear, either "horizontal"
                or "vertical".
            interpolation (str): Same as in :func:`mmcv.imshear`.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_sheared = mmcv.imshear(
                img,
                magnitude,
                direction,
                border_value=self.img_fill_val,
                interpolation=interpolation)
            results[key] = img_sheared.astype(img.dtype)
            results['img_shape'] = results[key].shape

    def _shear_bboxes(self, results, magnitude):
        """Shear the bboxes."""
        h, w, c = results['img_shape']
        if self.direction == 'horizontal':
            shear_matrix = np.stack([[1, magnitude],
                                     [0, 1]]).astype(np.float32)  # [2, 2]
        else:
            shear_matrix = np.stack([[1, 0], [magnitude,
                                              1]]).astype(np.float32)
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1)
            coordinates = np.stack([[min_x, min_y], [max_x, min_y],
                                    [min_x, max_y],
                                    [max_x, max_y]])  # [4, 2, nb_box, 1]
            coordinates = coordinates[..., 0].transpose(
                (2, 1, 0)).astype(np.float32)  # [nb_box, 2, 4]
            new_coords = np.matmul(shear_matrix[None, :, :],
                                   coordinates)  # [nb_box, 2, 4]
            min_x = np.min(new_coords[:, 0, :], axis=-1)
            min_y = np.min(new_coords[:, 1, :], axis=-1)
            max_x = np.max(new_coords[:, 0, :], axis=-1)
            max_y = np.max(new_coords[:, 1, :], axis=-1)
            min_x = np.clip(min_x, a_min=0, a_max=w)
            min_y = np.clip(min_y, a_min=0, a_max=h)
            max_x = np.clip(max_x, a_min=min_x, a_max=w)
            max_y = np.clip(max_y, a_min=min_y, a_max=h)
            results[key] = np.stack([min_x, min_y, max_x, max_y],
                                    axis=-1).astype(results[key].dtype)

    def _shear_masks(self,
                     results,
                     magnitude,
                     direction='horizontal',
                     fill_val=0,
                     interpolation='bilinear'):
        """Shear the masks."""
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.shear((h, w),
                                       magnitude,
                                       direction,
                                       border_value=fill_val,
                                       interpolation=interpolation)

    def _shear_seg(self,
                   results,
                   magnitude,
                   direction='horizontal',
                   fill_val=255,
                   interpolation='bilinear'):
        """Shear the segmentation maps."""
        for key in results.get('seg_fields', []):
            seg = results[key]
            results[key] = mmcv.imshear(
                seg,
                magnitude,
                direction,
                border_value=fill_val,
                interpolation=interpolation).astype(seg.dtype)

    def _filter_invalid(self, results, min_bbox_size=0):
        """Filter bboxes and corresponding masks too small after shear
        augmentation."""
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_inds = np.nonzero(valid_inds)[0]
            results[key] = results[key][valid_inds]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]

    def __call__(self, results):
        """Call function to shear images, bounding boxes, masks and semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Sheared results.
        """
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        self._shear_img(results, magnitude, self.direction, self.interpolation)
        self._shear_bboxes(results, magnitude)
        # fill_val set to 0 for background of mask.
        self._shear_masks(
            results,
            magnitude,
            self.direction,
            fill_val=0,
            interpolation=self.interpolation)
        self._shear_seg(
            results,
            magnitude,
            self.direction,
            fill_val=self.seg_ignore_label,
            interpolation=self.interpolation)
        self._filter_invalid(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'img_fill_val={self.img_fill_val}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'direction={self.direction}, '
        repr_str += f'max_shear_magnitude={self.max_shear_magnitude}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@PIPELINES.register_module()
class Rotate:
    """Apply Rotate Transformation to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        level (int | float): The level should be in range (0,_MAX_LEVEL].
        scale (int | float): Isotropic scale factor. Same in
            ``mmcv.imrotate``.
        center (int | float | tuple[float]): Center point (w, h) of the
            rotation in the source image. If None, the center of the
            image will be used. Same in ``mmcv.imrotate``.
        img_fill_val (int | float | tuple): The fill value for image border.
            If float, the same value will be used for all the three
            channels of image. If tuple, the should be 3 elements (e.g.
            equals the number of channels for image).
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        prob (float): The probability for perform transformation and
            should be in range 0 to 1.
        max_rotate_angle (int | float): The maximum angles for rotate
            transformation.
        random_negative_prob (float): The probability that turns the
             offset negative.
    """

    def __init__(self,
                 level,
                 scale=1,
                 center=None,
                 img_fill_val=128,
                 seg_ignore_label=255,
                 prob=0.5,
                 max_rotate_angle=30,
                 random_negative_prob=0.5):
        assert isinstance(level, (int, float)), \
            f'The level must be type int or float. got {type(level)}.'
        assert 0 <= level <= _MAX_LEVEL, \
            f'The level should be in range (0,{_MAX_LEVEL}]. got {level}.'
        assert isinstance(scale, (int, float)), \
            f'The scale must be type int or float. got type {type(scale)}.'
        if isinstance(center, (int, float)):
            center = (center, center)
        elif isinstance(center, tuple):
            assert len(center) == 2, 'center with type tuple must have '\
                f'2 elements. got {len(center)} elements.'
        else:
            assert center is None, 'center must be None or type int, '\
                f'float or tuple, got type {type(center)}.'
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, 'img_fill_val as tuple must '\
                f'have 3 elements. got {len(img_fill_val)}.'
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError(
                'img_fill_val must be float or tuple with 3 elements.')
        assert np.all([0 <= val <= 255 for val in img_fill_val]), \
            'all elements of img_fill_val should between range [0,255]. '\
            f'got {img_fill_val}.'
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. '\
            'got {prob}.'
        assert isinstance(max_rotate_angle, (int, float)), 'max_rotate_angle '\
            f'should be type int or float. got type {type(max_rotate_angle)}.'
        self.level = level
        self.scale = scale
        # Rotation angle in degrees. Positive values mean
        # clockwise rotation.
        self.angle = level_to_value(level, max_rotate_angle)
        # self.angle = 90
        self.center = center
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob
        self.max_rotate_angle = max_rotate_angle
        self.random_negative_prob = random_negative_prob

    def _rotate_img(self, results, angle, center=None, scale=1.0):
        """Rotate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            angle (float): Rotation angle in degrees, positive values
                mean clockwise rotation. Same in ``mmcv.imrotate``.
            center (tuple[float], optional): Center point (w, h) of the
                rotation. Same in ``mmcv.imrotate``.
            scale (int | float): Isotropic scale factor. Same in
                ``mmcv.imrotate``.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            img_rotated = mmcv.imrotate(
                img, angle, center, scale, border_value=self.img_fill_val)
            results[key] = img_rotated.astype(img.dtype)
            results['img_shape'] = results[key].shape

    def _rotate_bboxes(self, results, rotate_matrix):
        """Rotate the bboxes."""
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1)
            coordinates = np.stack([[min_x, min_y], [max_x, min_y],
                                    [min_x, max_y],
                                    [max_x, max_y]])  # [4, 2, nb_bbox, 1]
            # pad 1 to convert from format [x, y] to homogeneous
            # coordinates format [x, y, 1]
            coordinates = np.concatenate(
                (coordinates,
                 np.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype)),
                axis=1)  # [4, 3, nb_bbox, 1]
            coordinates = coordinates.transpose(
                (2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            rotated_coords = np.matmul(rotate_matrix,
                                       coordinates)  # [nb_bbox, 4, 2, 1]
            rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
            min_x, min_y = np.min(
                rotated_coords[:, :, 0], axis=1), np.min(
                    rotated_coords[:, :, 1], axis=1)
            max_x, max_y = np.max(
                rotated_coords[:, :, 0], axis=1), np.max(
                    rotated_coords[:, :, 1], axis=1)
            min_x, min_y = np.clip(
                min_x, a_min=0, a_max=w), np.clip(
                    min_y, a_min=0, a_max=h)
            max_x, max_y = np.clip(
                max_x, a_min=min_x, a_max=w), np.clip(
                    max_y, a_min=min_y, a_max=h)
            results[key] = np.stack([min_x, min_y, max_x, max_y],
                                    axis=-1).astype(results[key].dtype)

    def _rotate_masks(self,
                      results,
                      angle,
                      center=None,
                      scale=1.0,
                      fill_val=0):
        """Rotate the masks."""
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.rotate((h, w), angle, center, scale, fill_val)

    def _rotate_seg(self,
                    results,
                    angle,
                    center=None,
                    scale=1.0,
                    fill_val=255):
        """Rotate the segmentation map."""
        for key in results.get('seg_fields', []):
            seg = results[key].copy()
            results[key] = mmcv.imrotate(
                seg, angle, center, scale,
                border_value=fill_val).astype(seg.dtype)

    def _filter_invalid(self, results, min_bbox_size=0):
        """Filter bboxes and corresponding masks too small after rotate
        augmentation."""
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_inds = np.nonzero(valid_inds)[0]
            results[key] = results[key][valid_inds]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]

    def __call__(self, results):
        """Call function to rotate images, bounding boxes, masks and semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        if np.random.rand() > self.prob:
            return results
        h, w = results['img'].shape[:2]
        center = self.center
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        angle = random_negative(self.angle, self.random_negative_prob)
        self._rotate_img(results, angle, center, self.scale)
        rotate_matrix = cv2.getRotationMatrix2D(center, -angle, self.scale)
        self._rotate_bboxes(results, rotate_matrix)
        self._rotate_masks(results, angle, center, self.scale, fill_val=0)
        self._rotate_seg(
            results, angle, center, self.scale, fill_val=self.seg_ignore_label)
        self._filter_invalid(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'center={self.center}, '
        repr_str += f'img_fill_val={self.img_fill_val}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'max_rotate_angle={self.max_rotate_angle}, '
        repr_str += f'random_negative_prob={self.random_negative_prob})'
        return repr_str


@PIPELINES.register_module()
class Translate:
    """Translate the images, bboxes, masks and segmentation maps horizontally
    or vertically.

    Args:
        level (int | float): The level for Translate and should be in
            range [0,_MAX_LEVEL].
        prob (float): The probability for performing translation and
            should be in range [0, 1].
        img_fill_val (int | float | tuple): The filled value for image
            border. If float, the same fill value will be used for all
            the three channels of image. If tuple, the should be 3
            elements (e.g. equals the number of channels for image).
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        direction (str): The translate direction, either "horizontal"
            or "vertical".
        max_translate_offset (int | float): The maximum pixel's offset for
            Translate.
        random_negative_prob (float): The probability that turns the
            offset negative.
        min_size (int | float): The minimum pixel for filtering
            invalid bboxes after the translation.
    """

    def __init__(self,
                 level,
                 prob=0.5,
                 img_fill_val=128,
                 seg_ignore_label=255,
                 direction='horizontal',
                 max_translate_offset=250.,
                 random_negative_prob=0.5,
                 min_size=0):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level used for calculating Translate\'s offset should be ' \
            'in range [0,_MAX_LEVEL]'
        assert 0 <= prob <= 1.0, \
            'The probability of translation should be in range [0, 1].'
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, \
                'img_fill_val as tuple must have 3 elements.'
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError('img_fill_val must be type float or tuple.')
        assert np.all([0 <= val <= 255 for val in img_fill_val]), \
            'all elements of img_fill_val should between range [0,255].'
        assert direction in ('horizontal', 'vertical'), \
            'direction should be "horizontal" or "vertical".'
        assert isinstance(max_translate_offset, (int, float)), \
            'The max_translate_offset must be type int or float.'
        # the offset used for translation
        self.offset = int(level_to_value(level, max_translate_offset))
        self.level = level
        self.prob = prob
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.direction = direction
        self.max_translate_offset = max_translate_offset
        self.random_negative_prob = random_negative_prob
        self.min_size = min_size

    def _translate_img(self, results, offset, direction='horizontal'):
        """Translate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
        """
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            results[key] = mmcv.imtranslate(
                img, offset, direction, self.img_fill_val).astype(img.dtype)
            results['img_shape'] = results[key].shape

    def _translate_bboxes(self, results, offset):
        """Shift bboxes horizontally or vertically, according to offset."""
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1)
            if self.direction == 'horizontal':
                min_x = np.maximum(0, min_x + offset)
                max_x = np.minimum(w, max_x + offset)
            elif self.direction == 'vertical':
                min_y = np.maximum(0, min_y + offset)
                max_y = np.minimum(h, max_y + offset)

            # the boxes translated outside of image will be filtered along with
            # the corresponding masks, by invoking ``_filter_invalid``.
            results[key] = np.concatenate([min_x, min_y, max_x, max_y],
                                          axis=-1)

    def _translate_masks(self,
                         results,
                         offset,
                         direction='horizontal',
                         fill_val=0):
        """Translate masks horizontally or vertically."""
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.translate((h, w), offset, direction, fill_val)

    def _translate_seg(self,
                       results,
                       offset,
                       direction='horizontal',
                       fill_val=255):
        """Translate segmentation maps horizontally or vertically."""
        for key in results.get('seg_fields', []):
            seg = results[key].copy()
            results[key] = mmcv.imtranslate(seg, offset, direction,
                                            fill_val).astype(seg.dtype)

    def _filter_invalid(self, results, min_size=0):
        """Filter bboxes and masks too small or translated out of image."""
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_size) & (bbox_h > min_size)
            valid_inds = np.nonzero(valid_inds)[0]
            results[key] = results[key][valid_inds]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]
        return results

    def __call__(self, results):
        """Call function to translate images, bounding boxes, masks and
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Translated results.
        """
        if np.random.rand() > self.prob:
            return results
        offset = random_negative(self.offset, self.random_negative_prob)
        self._translate_img(results, offset, self.direction)
        self._translate_bboxes(results, offset)
        # fill_val defaultly 0 for BitmapMasks and None for PolygonMasks.
        self._translate_masks(results, offset, self.direction)
        # fill_val set to ``seg_ignore_label`` for the ignored value
        # of segmentation map.
        self._translate_seg(
            results, offset, self.direction, fill_val=self.seg_ignore_label)
        self._filter_invalid(results, min_size=self.min_size)
        return results


@PIPELINES.register_module()
class ColorTransform:
    """Apply Color transformation to image. The bboxes, masks, and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Color transformation.
    """

    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _adjust_color_img(self, results, factor=1.0):
        """Apply Color transformation to image."""
        for key in results.get('img_fields', ['img']):
            # NOTE defaultly the image should be BGR format
            img = results[key]
            results[key] = mmcv.adjust_color(img, factor).astype(img.dtype)

    def __call__(self, results):
        """Call function for Color transformation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Colored results.
        """
        if np.random.rand() > self.prob:
            return results
        self._adjust_color_img(results, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class EqualizeTransform:
    """Apply Equalize transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        prob (float): The probability for performing Equalize transformation.
    """

    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.prob = prob

    def _imequalize(self, results):
        """Equalizes the histogram of one image."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = mmcv.imequalize(img).astype(img.dtype)

    def __call__(self, results):
        """Call function for Equalize transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._imequalize(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'


@PIPELINES.register_module()
class BrightnessTransform:
    """Apply Brightness transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Brightness transformation.
    """

    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _adjust_brightness_img(self, results, factor=1.0):
        """Adjust the brightness of image."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = mmcv.adjust_brightness(img,
                                                  factor).astype(img.dtype)

    def __call__(self, results):
        """Call function for Brightness transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._adjust_brightness_img(results, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class ContrastTransform:
    """Apply Contrast transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Contrast transformation.
    """

    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _adjust_contrast_img(self, results, factor=1.0):
        """Adjust the image contrast."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = mmcv.adjust_contrast(img, factor).astype(img.dtype)

    def __call__(self, results):
        """Call function for Contrast transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._adjust_contrast_img(results, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class AutoContrastTransform:
    """Apply Contrast transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Contrast transformation.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def _scale_channel(self, image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = float(np.min(image))
        hi = float(np.max(image))

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = im.astype(np.float32) * scale + offset
            img = np.clip(im, a_min=0, a_max=255.0)
            return img.astype(np.uint8)

        result = scale_values(image) if hi > lo else image
        return result

    def __call__(self, results):
        """Call function for Contrast transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
            # Assumes RGB for now.    Scales each channel independently
            # and then stacks the result.
        for key in results.get('img_fields', ['img']):
            img = results[key]
            s1 = self._scale_channel(img[:, :, 0])
            s2 = self._scale_channel(img[:, :, 1])
            s3 = self._scale_channel(img[:, :, 2])
            results[key] = np.stack([s1, s2, s3], 2)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class SolarizeAdd:
    """Apply Contrast transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Contrast transformation.
    """

    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = int((level / _MAX_LEVEL) * 110)

    def _solarize_add(self, image, addition=0, threshold=128):
        # For each pixel in the image less than threshold
        # we add 'addition' amount to it and then clip the
        # pixel value to be between 0 and 255. The value
        # of 'addition' is between -128 and 128.
        added_image = image.astype(np.int64) + addition
        added_image = np.clip(added_image, a_min=0, a_max=255).astype(np.uint8)
        return np.where(image < threshold, added_image, image)

    def __call__(self, results):
        """Call function for Contrast transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
            # Assumes RGB for now.    Scales each channel independently
            # and then stacks the result.
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = self._solarize_add(img, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class PosterSize:
    """Apply Contrast transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Contrast transformation.
    """

    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = int((level / _MAX_LEVEL) * 4)

    def _posterize(self, image, bits):
        """Equivalent of PIL Posterize."""
        shift = 8 - bits
        return np.left_shift(np.right_shift(image, shift), shift)

    def __call__(self, results):
        """Call function for Contrast transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
            # Assumes RGB for now.    Scales each channel independently
            # and then stacks the result.
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = self._posterize(img, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class BBox_Cutout:
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Args:
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
    """

    def __init__(self,
                 level,
                 prob=0.5):
        self.factor = level / _MAX_LEVEL * 0.75
        self.prob = prob

    def _cutout_inside_bbox(self, image, bbox, pad_fraction):
        """Generates cutout mask and the mean pixel value of the bbox.

        First a location is randomly chosen within the image as the center where the
        cutout mask will be applied. Note this can be towards the boundaries of the
        image, so the full cutout mask may not be applied.

        Args:
            image: 3D uint8 Tensor.
            bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
                of type float that represents the normalized coordinates between 0 and 1.
            pad_fraction: Float that specifies how large the cutout mask should be in
                in reference to the size of the original bbox. If pad_fraction is 0.25,
                then the cutout mask will be of shape
                (0.25 * bbox height, 0.25 * bbox width).

        Returns:
            A tuple. Fist element is a tensor of the same shape as image where each
            element is either a 1 or 0 that is used to determine where the image
            will have cutout applied. The second element is the mean of the pixels
            in the image where the bbox is located.
            mask value: [0,1]
        """
        image_height, image_width = image.shape[0], image.shape[1]
        # Transform from shape [1, 4] to [4].
        bbox = np.squeeze(bbox)

        min_x = int(bbox[0])
        min_y = int(bbox[1])
        max_x = int(bbox[2])
        max_y = int(bbox[3])

        # Calculate the mean pixel values in the bounding box, which will be used
        # to fill the cutout region.
        mean = np.mean(image[min_y:max_y + 1, min_x:max_x + 1], axis=(0, 1))
        # Cutout mask will be size pad_size_heigh * 2 by pad_size_width * 2 if the
        # region lies entirely within the bbox.
        box_height = max_y - min_y + 1
        box_width = max_x - min_x + 1
        pad_size_height = int(pad_fraction * (box_height / 2))
        pad_size_width = int(pad_fraction * (box_width / 2))

        # Sample the center location in the image where the zero mask will be applied.
        cutout_center_height = np.random.randint(min_y, max_y + 1, dtype=np.int32)
        cutout_center_width = np.random.randint(min_x, max_x + 1, dtype=np.int32)

        lower_pad = np.maximum(0, cutout_center_height - pad_size_height)
        upper_pad = np.maximum(
            0, image_height - cutout_center_height - pad_size_height)
        left_pad = np.maximum(0, cutout_center_width - pad_size_width)
        right_pad = np.maximum(0,
                               image_width - cutout_center_width - pad_size_width)

        cutout_shape = [
            image_height - (lower_pad + upper_pad),
            image_width - (left_pad + right_pad)
        ]
        padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]

        mask = np.pad(np.zeros(
            cutout_shape, dtype=image.dtype),
            padding_dims,
            'constant',
            constant_values=1)

        mask = np.expand_dims(mask, 2)
        mask = np.tile(mask, [1, 1, 3])
        return mask, mean

    def _bbox_cutout(self, image, bboxes, pad_fraction, replace_with_mean):
        """Applies cutout to the image according to bbox information.

        This is a cutout variant that using bbox information to make more informed
        decisions on where to place the cutout mask.

        Args:
            image: 3D uint8 Tensor.
            bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
                has 4 elements (min_y, min_x, max_y, max_x) of type float with values
                between [0, 1].
            pad_fraction: Float that specifies how large the cutout mask should be in
                in reference to the size of the original bbox. If pad_fraction is 0.25,
                then the cutout mask will be of shape
                (0.25 * bbox height, 0.25 * bbox width).
            replace_with_mean: Boolean that specified what value should be filled in
                where the cutout mask is applied. Since the incoming image will be of
                uint8 and will not have had any mean normalization applied, by default
                we set the value to be 128. If replace_with_mean is True then we find
                the mean pixel values across the channel dimension and use those to fill
                in where the cutout mask is applied.

        Returns:
            A tuple. First element is a tensor of the same shape as image that has
            cutout applied to it. Second element is the bboxes that were passed in
            that will be unchanged.
        """

        def apply_bbox_cutout(image, bboxes, pad_fraction):
            """Applies cutout to a single bounding box within image."""
            # Choose a single bounding box to apply cutout to.
            random_index = np.random.randint(0, bboxes.shape[0], dtype=np.int32)
            # Select the corresponding bbox and apply cutout.
            chosen_bbox = np.take(bboxes, random_index, axis=0)
            mask, mean = self._cutout_inside_bbox(image, chosen_bbox, pad_fraction)

            # When applying cutout we either set the pixel value to 128 or to the mean
            # value inside the bbox.
            replace = mean if replace_with_mean else [128] * 3

            # Apply the cutout mask to the image. Where the mask is 0 we fill it with
            # `replace`.
            image = np.where(
                np.equal(mask, 0),
                np.ones_like(
                    image, dtype=image.dtype) * replace,
                image).astype(image.dtype)
            return image

        # Check to see if there are boxes, if so then apply boxcutout.
        if len(bboxes) != 0:
            image = apply_bbox_cutout(image, bboxes, pad_fraction)

        return image, bboxes

    def __call__(self, results):
        """Call function to drop some regions of image."""
        if np.random.rand() > self.prob:
            return results
        image = results['img']
        bboxes = results['gt_bboxes']
        image, bboxes = self._bbox_cutout(image, bboxes, self.factor, replace_with_mean=False)
        results['img'] = image
        results['gt_bboxes'] = bboxes
        return results


@PIPELINES.register_module()
class Cutout_AA:
    """Apply Contrast transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Contrast transformation.
    """

    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = int((level / _MAX_LEVEL) * 100)

    def _cutout(self, image, pad_size, replace=0):
        """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

        This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
        a random location within `img`. The pixel values filled in will be of the
        value `replace`. The located where the mask will be applied is randomly
        chosen uniformly over the whole image.

        Args:
            image: An image Tensor of type uint8.
            pad_size: Specifies how big the zero mask that will be generated is that
                is applied to the image. The mask will be of size
                (2*pad_size x 2*pad_size).
            replace: What pixel value to fill in the image in the area that has
                the cutout mask applied to it.

        Returns:
            An image Tensor that is of type uint8.
        Example:
            img = cv2.imread( "/home/vis/gry/train/img_data/test.jpg", cv2.COLOR_BGR2RGB )
            new_img = cutout(img, pad_size=50, replace=0)
        """
        image_height, image_width = image.shape[0], image.shape[1]

        cutout_center_height = np.random.randint(low=0, high=image_height)
        cutout_center_width = np.random.randint(low=0, high=image_width)

        lower_pad = np.maximum(0, cutout_center_height - pad_size)
        upper_pad = np.maximum(0, image_height - cutout_center_height - pad_size)
        left_pad = np.maximum(0, cutout_center_width - pad_size)
        right_pad = np.maximum(0, image_width - cutout_center_width - pad_size)

        cutout_shape = [
            image_height - (lower_pad + upper_pad),
            image_width - (left_pad + right_pad)
        ]
        padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
        mask = np.pad(np.zeros(
            cutout_shape, dtype=image.dtype),
            padding_dims,
            'constant',
            constant_values=1)
        mask = np.expand_dims(mask, -1)
        mask = np.tile(mask, [1, 1, 3])
        image = np.where(
            np.equal(mask, 0),
            np.ones_like(
                image, dtype=image.dtype) * replace,
            image)
        return image.astype(np.uint8)

    def __call__(self, results):
        """Call function for Contrast transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
            # Assumes RGB for now.    Scales each channel independently
            # and then stacks the result.
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = self._cutout(img, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class Sharpness:
    """Apply Contrast transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Contrast transformation.
    """

    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _blend(self, image1, image2, factor):
        """Blend image1 and image2 using 'factor'.

        Factor can be above 0.0.    A value of 0.0 means only image1 is used.
        A value of 1.0 means only image2 is used.    A value between 0.0 and
        1.0 means we linearly interpolate the pixel values between the two
        images.    A value greater than 1.0 "extrapolates" the difference
        between the two pixel values, and we clip the results to values
        between 0 and 255.

        Args:
            image1: An image Tensor of type uint8.
            image2: An image Tensor of type uint8.
            factor: A floating point value above 0.0.

        Returns:
            A blended image Tensor of type uint8.
        """
        if factor == 0.0:
            return image1
        if factor == 1.0:
            return image2

        image1 = image1.astype(np.float32)
        image2 = image2.astype(np.float32)

        difference = image2 - image1
        scaled = factor * difference

        # Do addition in float.
        temp = image1 + scaled

        # Interpolate
        if factor > 0.0 and factor < 1.0:
            # Interpolation means we always stay within 0 and 255.
            return temp.astype(np.uint8)

        # Extrapolate:
        #
        # We need to clip and then cast.
        return np.clip(temp, a_min=0, a_max=255).astype(np.uint8)

    def _sharpness(self, image, factor):
        """Implements Sharpness function from PIL."""
        orig_image = image
        image = image.astype(np.float32)
        # Make image 4D for conv operation.
        # SMOOTH PIL Kernel.
        kernel = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32) / 13.
        result = cv2.filter2D(image, -1, kernel).astype(np.uint8)

        # Blend the final result.
        return self._blend(result, orig_image, factor)

    def __call__(self, results):
        """Call function for Contrast transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
            # Assumes RGB for now.    Scales each channel independently
            # and then stacks the result.
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = self._sharpness(img, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str