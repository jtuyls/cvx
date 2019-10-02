"""
Module for testing image processing operations

Authors: Jorn Tuyls
"""

import unittest
import numpy as np

import sys
import logging
logger = logging.getLogger('cvx')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)

from cvx import ops

class TestImgProcessor(unittest.TestCase):

    def test_crop(self):
        logger.debug("Test crop")
        
        # HWC
        img = np.transpose(
            np.reshape(np.array([
                [[10,10,0],
                [50,10,0],
                [0,0,0]],
                [[30,50,0],
                [10,90,0],
                [0,0,0]],
                [[20,0,0],
                [0,0,0],
                [0,0,0]]
            ], np.float32), (3,3,3)),
            (1,2,0)
        )

        crop_func = ops.crop(height=[0,2], width=[0,2], channels=[1,3])
        res = crop_func(img)

        expected_outpt = np.transpose(
            np.reshape(np.array([
                [[30,50],
                [10,90]],
                [[20, 0],
                [0, 0]]
            ]), (2,2,2)),
            (1,2,0)
        ) # HWC
        
        np.testing.assert_array_equal(res, expected_outpt)

    def test_normalize(self):
        logger.debug("Test normalize")
        
        # HWC
        img = np.transpose(
            np.reshape(np.array([
                [[10,10],
                [50,10]],
                [[30,50],
                [10,90]],
                [[20, 0],
                [0, 0]]
            ], np.float32), (3,2,2)),
            (1,2,0)
        )

        means, stdevs = [30,30,10], [2,1.5,1]
        norm_func = ops.normalize(means=means, stdevs=stdevs)
        res = norm_func(img)

        expected_outpt = (img - means) / stdevs # HWC
        
        np.testing.assert_array_equal(res, expected_outpt)

    def test_resize(self):
        logger.debug("Test resize")
        
        # HWC
        img = np.transpose(
            np.reshape(np.array([
                [[10,10,0],
                [50,10,0],
                [0,0,0]],
                [[30,50,0],
                [10,90,0],
                [0,0,0]],
                [[20,0,0],
                [0,0,0],
                [0,0,0]]
            ], np.float32), (3,3,3)),
            (1,2,0)
        )

        resize_func = ops.resize(size=[3,2])
        res = resize_func(img)

        assert(len(res.shape) == 3 and res.shape[0] == 2 and res.shape[1] == 3)

    def test_resize_aspect_preservation(self):
        logger.debug("Test resize aspect preservation")
        
        # HWC
        img = np.transpose(
            np.reshape(np.array([
                [[10,10,0],
                [50,10,0],
                [0,0,0]],
                [[30,50,0],
                [10,90,0],
                [0,0,0]],
                [[20,0,0],
                [0,0,0],
                [0,0,0]]
            ], np.float32), (3,3,3)),
            (1,2,0)
        )

        resize_func = ops.resize(size=[2,None])
        res = resize_func(img)
        assert(len(res.shape) == 3 and res.shape[0] == 2 and res.shape[1] == 2)

        resize_func = ops.resize(size=[1,None])
        res = resize_func(img)
        assert(len(res.shape) == 3 and res.shape[0] == 1 and res.shape[1] == 1)

        resize_func = ops.resize(size=[3,None])
        res = resize_func(img)
        assert(len(res.shape) == 3 and res.shape[0] == 3 and res.shape[1] == 3)

        # HWC
        img = np.transpose(
            np.reshape(np.array([
                [[10,10,0],
                [50,10,0],
                [0,0,0],
                [0,0,0]],
                [[30,50,0],
                [10,90,0],
                [0,0,0],
                [0,0,0]],
                [[20,0,0],
                [0,0,0],
                [0,0,0],
                [0,0,0]]
            ], np.float32), (3,4,3)),
            (1,2,0)
        )

        resize_func = ops.resize(size=[None,2])
        res = resize_func(img)
        assert(len(res.shape) == 3 and res.shape[0] == 2 and res.shape[1] == 1)

    def test_scale(self):
        logger.debug("Test scale")
        
        # HWC
        img = np.transpose(
            np.reshape(np.array([
                [[10,10],
                [50,10]],
                [[30,50],
                [10,90]],
                [[20, 0],
                [0, 0]]
            ], np.float32), (3,2,2)),
            (1,2,0)
        )

        scale_func = ops.scale(scale=1.5)
        res = scale_func(img)

        expected_outpt = 1.5 * img # HWC
        
        np.testing.assert_array_equal(res, expected_outpt)

    def test_subtract(self):
        logger.debug("Test subtract")
        
        # HWC
        img = np.transpose(
            np.reshape(np.array([
                [[10,10],
                [50,10]],
                [[30,50],
                [10,90]],
                [[20, 0],
                [0, 0]]
            ], np.float32), (3,2,2)),
            (1,2,0)
        )

        means = [30, 30, 10]
        subtract_func = ops.subtract(values=means)
        res = subtract_func(img)

        expected_outpt = img - means
        
        np.testing.assert_array_equal(res, expected_outpt)

    def test_transpose(self):
        logger.debug("Test transpose")
        
        # HWC
        img = np.transpose(
            np.reshape(np.array([
                [[10,10],
                [50,10]],
                [[30,50],
                [10,90]],
                [[20, 0],
                [0, 0]]
            ], np.float32), (3,2,2)),
            (1,2,0)
        )

        transpose_func = ops.transpose(axes=[2,0,1])
        res = transpose_func(img)

        expected_outpt = np.transpose(img, axes=(2,0,1))
        
        np.testing.assert_array_equal(res, expected_outpt)


if __name__ == '__main__':
    unittest.main()