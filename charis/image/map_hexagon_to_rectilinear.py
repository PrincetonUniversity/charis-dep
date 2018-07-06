#!/usr/bin/env python

from __future__ import print_function, absolute_import

"""
Utilities for remapping hexagonal grid output to rectilinear (fishnet)
grid. Partially based on http://www.redblobgames.com/grids/hexagons/

Based on Sutherland-Hodgman algorithm for computing the cross section
between arbitrary polygons.

Experimental version
"""


import os
import json
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

import collections
import math

from tqdm import tqdm

from itertools import product
try:
    from sutherland_hodgman import clip, area
except:
    from .sutherland_hodgman import clip, area

Point = collections.namedtuple("Point", ["x", "y"])
_Hex = collections.namedtuple("Hex", ["q", "r", "s"])
Orientation = collections.namedtuple("Orientation", ["f0", "f1", "f2", "f3", "b0", "b1", "b2", "b3", "start_angle"])
Layout = collections.namedtuple("Layout", ["orientation", "size", "origin"])
Layout_square = collections.namedtuple("Layout_square", ["size", "origin"])

layout_pointy = Orientation(np.sqrt(3.0), np.sqrt(3.0) / 2.0, 0.0, 3.0 / 2.0,
                            np.sqrt(3.0) / 3.0, -1.0 / 3.0, 0.0, 2.0 / 3.0, 0.5)
layout_flat = Orientation(3.0 / 2.0, 0.0, np.sqrt(3.0) / 2.0, np.sqrt(3.0),
                          2.0 / 3.0, 0.0, -1.0 / 3.0, np.sqrt(3.0) / 3.0, 0.0)


def hex_corner_offset(layout, corner):
    M = layout.orientation
    size = layout.size
    angle = 2.0 * np.pi * (M.start_angle - corner) / 6
    return Point(size.x * np.cos(angle), size.y * np.sin(angle))


def polygon_corners(layout, center):
    origin = layout.origin
    corners = []
    for i in range(0, 6):
        offset = hex_corner_offset(layout, i)
        corners.append(
            [center.x + offset.x + origin.x,
             center.y + offset.y + origin.y])
    return corners


def square_corner_offset(size, theta):
    x = np.sqrt(2) * size / 2 * np.cos(theta)
    y = np.sqrt(2) * size / 2 * np.sin(theta)
    return Point(x, y)


def square_corners(layout, center):
    corners = []
    origin = layout.origin
    size = layout.size[0]
    angles = [np.pi / 4, np.pi * 3 / 4, np.pi * 5 / 4, np.pi * 7 / 4]
    for angle in angles:
        offset = square_corner_offset(size, angle)
        corners.append(
            [center.x + offset.x + origin.x,
             center.y + offset.y + origin.y])
    return corners


def flatten_cube(image_cube):
    number_of_frames = image_cube.shape[0]
    number_of_pixels = image_cube.shape[-1]**2
    return image_cube.reshape(number_of_frames, number_of_pixels)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# corners_hex = np.array(polygon_corners(pointy, Point(10, 10)))
# corners_square = np.array(square_corners(square, Point(10, 10)))
# # corners_hex = polygon_corners(pointy, Point(10, 10))
# # corners_square = square_corners(square, Point(10, 10))
#
# clipped = clip(corners_hex, corners_square)
# poly_area = area(clipped)

# plt.plot(corners_hex[:, 0], corners_hex[:, 1])
# plt.plot(corners_square[:, 0], corners_square[:, 1])
# plt.show()
# plt.gca().set_aspect('equal')
# plt.scatter(corners[:, 0], corners[:, 1])
# plt.show()

# Fake hexagon plot
# norm_counts = ImageNormalize(flat_cube[12], interval=ZScaleInterval())
# plt.scatter(hexagon_centers[:,0], hexagon_centers[:,1], c=flat_cube[12], norm=norm_counts)
# plt.gca().set_aspect('equal')
# plt.show()

# plt.close()
# for i in tqdm(range(len(hexagon_arr) // 16)):
#     plt.plot(hexagon_arr[i, :, 0], hexagon_arr[i, :, 1])  # , label=str(i))
# for i in tqdm(range(len(square_arr) // 16)):
#     plt.plot(square_arr[i, :, 0], square_arr[i, :, 1])
#
# # plt.scatter(squares[:, 0], squares[:, 1])
# # plt.legend()
# plt.gca().set_aspect('equal')
# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.show()

# a_list = clip(hexagon_arr[0].tolist(), square_arr[0].tolist())
# a = np.array(a_list)
# plt.plot(a[:, 0] + 0.02, a[:, 1] + 0.02)
# plt.plot(hexagon_arr[0, :, 0], hexagon_arr[0, :, 1])  # , label=str(i))
# plt.plot(square_arr[0, :, 0], square_arr[0, :, 1])
# plt.show()
# area_clip = area(a_list)
# area_hex = area(hexagon_arr[0].tolist())


def contribution_by_hexagons(square_center, square_corners, hexagon_centers, hexagon_arr, dmax):
    distances = np.linalg.norm(square_center - hexagon_centers, axis=1)
    mask = distances < 4 * dmax
    hexagon_indices = np.where(mask)
    hexagons = hexagon_arr[mask]
    clip_polygons = [clip(hexagon.tolist(), square_corners.tolist()) for hexagon in hexagons]

    empty_overlap = []
    for idx, polygon in enumerate(clip_polygons):
        if len(polygon) == 0:
            empty_overlap.append(idx)
    if len(empty_overlap) > 0:
        clip_polygons = np.delete(np.array(clip_polygons), empty_overlap, axis=0).tolist()
        hexagon_indices = np.delete(hexagon_indices, empty_overlap).tolist()
    areas = [area(polygon) for polygon in clip_polygons]
    polygon_clipped = {  # 'polygon': clip_polygons,
        'hex_indices': list(hexagon_indices),
        'areas': list(areas)}
    return polygon_clipped


def make_mapping(image_size=201, oversampling=2,
                 outputname=None):

    hexagon_size = 1. / np.sqrt(3)
    square_size = hexagon_size / oversampling * 3**(0.75)
    dmax = max(hexagon_size, square_size)

    square = Layout_square(size=Point(square_size, square_size), origin=Point(0., 0.))
    pointy = Layout(layout_pointy, size=Point(hexagon_size, hexagon_size), origin=Point(0., 0.))
    flat = Layout(layout_flat, size=Point(1, 1), origin=Point(0., 0.))

    x = np.arange(0, image_size) * 1.
    X, Y = np.meshgrid(x, x)
    X[::2] -= 0.5
    Y *= np.sqrt(3) / 2
    hexagon_centers = np.vstack([X.ravel(), Y.ravel()]).T

    x_square = np.arange(np.min((X, Y)), np.max((X, Y)), square_size)
    y_square = np.arange(np.min((X, Y)), np.max((X, Y)), square_size)

    square_centers = np.array(list(product(x_square, y_square)))

    hexagons = []
    for hexagon_center in hexagon_centers:  # product(x_hexagon, y_hexagon):
        # print(hexagon_center)
        hexagons += polygon_corners(pointy, Point(hexagon_center[0], hexagon_center[1]))
    squares = []
    for square_center in square_centers:
        squares += square_corners(square, Point(square_center[0], square_center[1]))

    squares = np.array(squares)
    square_arr = squares.reshape(len(x_square)**2, 4, 2)
    hexagons = np.array(hexagons)
    hexagon_arr = hexagons.reshape(image_size**2, 6, 2)

    clip_infos = []
    for i in tqdm(range(len(square_arr))):
        clip_infos.append(contribution_by_hexagons(
            square_center=square_centers[i],
            square_corners=square_arr[i],
            hexagon_centers=hexagon_centers,
            hexagon_arr=hexagon_arr,
            dmax=dmax))
    if outputname is not None:
        with open(outputname, 'w') as fout:
            json.dump(clip_infos, fout, cls=NumpyEncoder)

    return clip_infos


# clip_infos = make_mapping(image_size=201, oversampling=2,
#                           outputname=None)
# clip_infos = make_mapping(square_center, square_corners, hexagon_centers,
#                           hexagon_arr, dmax,
#                           'mapping_calib.pickle')


def resample_image_cube(image_cube, clip_infos, hexagon_size=1 / np.sqrt(3)):
    flat_cube = flatten_cube(image_cube)
    hexagon_area = hexagon_size**2 * 3 / 2 * np.sqrt(3)
    image_cube = np.zeros([len(flat_cube), len(clip_infos)])
    number_of_pixels = int(np.sqrt(len(clip_infos)))
    for wave in tqdm(range(len(image_cube))):
        for pixel, clip_info in enumerate(clip_infos):
            if len(clip_info['areas']) > 0:
                image_cube[wave, pixel] = np.sum(
                    flat_cube[wave][clip_info['hex_indices']] *
                    np.array(clip_info['areas']) / hexagon_area)
    image_cube = image_cube.reshape(
        len(image_cube), number_of_pixels, number_of_pixels)
    image_cube = np.swapaxes(image_cube, 1, 2)
    return image_cube


def resample_image_cube_file(filename, clip_info_file, hexagon_size=1 / np.sqrt(3)):
    clip_infos = pickle.load(open(clip_info_file, "rb"))

    image_cube = fits.getdata(filename)
    header = fits.getheader(filename)

    image_cube = resample_image_cube(
        image_cube, clip_infos, hexagon_size=1 / np.sqrt(3))

    outputfilename = os.path.splitext(filename)[0] + '_resampled.fits'
    fits.writeto(outputfilename, image_cube, header, overwrite=True)

# for polygon in clip_polygons:
#     polygon = np.array(polygon)
#     plt.plot(polygon[:, 0]+0.1, polygon[:, 1]+0.1)

# plt.show()
# plt.scatter(hexagons[:, 0], hexagons[:, 1])
# plt.scatter(squares[:, 0], squares[:, 1])
# # plt.scatter(corners_hex[:,0], corners_hex[:,1])
# plt.gca().set_aspect('equal')
# # plt.xlim(0, 51)
# # plt.ylim(0, 51)
# plt.show()
