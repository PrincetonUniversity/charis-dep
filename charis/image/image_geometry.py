#!/usr/bin/env python

from __future__ import print_function, absolute_import

"""
Utilities for remapping hexagonal grid output to rectilinear (fishnet)
grid. Partially based on http://www.redblobgames.com/grids/hexagons/

Based on Sutherland-Hodgman algorithm for computing the cross section
between arbitrary polygons.

"""


import os
import json
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

from astropy.stats import sigma_clip, mad_std
from tqdm import tqdm
import collections
import math
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


def deflatten_cube(flat_cube):
    number_of_frames = flat_cube.shape[0]
    image_size = int(np.sqrt(flat_cube.shape[-1]))
    return flat_cube.reshape(number_of_frames, image_size, image_size)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def make_hex_coordinates(image_size, hexagon_size=1. / np.sqrt(3), layout='pointy'):
    """Creates array containing the center of each hexagon

    Parameters
    ----------
    image_size : int
        Number of lenslets in one dimension.
    hexagon_size : float
        Side length of hexagon.
    layout : str
        'pointy' or 'square' hexagon layout.

    Returns
    -------
    type
        Array containing the center of each hexagone.

    """

    x = np.arange(0, image_size) * hexagon_size * np.sqrt(3)
    if layout == 'pointy':
        X, Y = np.meshgrid(x, x)
        X[::2] -= hexagon_size * np.sqrt(3) / 2.
        Y *= np.sqrt(3) / 2.
    elif layout == 'flat':
        Y, X = np.meshgrid(x, x)
        X *= np.sqrt(3) / 2.
        Y[::2] -= hexagon_size * np.sqrt(3) / 2.
    else:
        raise ValueError('Only pointy and flat geometry available.')

    hexagon_centers = np.vstack([X.ravel(), Y.ravel()]).T

    return hexagon_centers


def make_hexagon_array(image_size, hexagon_size=1. / np.sqrt(3),
                       layout='pointy'):
    """Create hexagon array

    Parameters
    ----------
    image_size : int
        Number of lenslets in one dimension.
    hexagon_size : float
        Side length of hexagon.
    layout : str
        'pointy' or 'square' hexagon layout.

    Returns
    -------
    type
        Array containing coordinates of corners for each lenslet.

    """

    if layout == 'pointy':
        geometry = Layout(
            layout_pointy, size=Point(hexagon_size, hexagon_size),
            origin=Point(0., 0.))
    elif layout == 'flat':
        geometry = Layout(
            layout_flat, size=Point(hexagon_size, hexagon_size),
            origin=Point(0., 0.))
    else:
        raise ValueError('Only pointy and flat geometry available.')

    hexagon_centers = make_hex_coordinates(
        image_size, hexagon_size, layout)
    hexagons = []
    for hexagon_center in hexagon_centers:
        hexagons += polygon_corners(geometry, Point(hexagon_center[0], hexagon_center[1]))
    hexagons = np.array(hexagons)
    hexagon_array = hexagons.reshape(image_size**2, 6, 2)

    return hexagon_array


def make_pixel_overlay_grid(image_size, hexagon_size=1. / np.sqrt(3),
                            square_size=1. / np.sqrt(3), layout='pointy'):
    """Create rectilinear grid covering the lenslet grid.
       Used as input for mapping routine.

    Parameters
    ----------
    image_size : int
        Number of lenslets in one dimension.
    hexagon_size : float
        Side length of hexagon.
    square_size : float
        Side length of pixel.
    layout : str
        'pointy' or 'square' hexagon layout.


    Returns
    -------
    tuple
        Array containing coordinates of centers
        and array containing coordinates of corners for each pixel.

    """

    hexagon_centers = make_hex_coordinates(
        image_size, hexagon_size, layout)
    square = Layout_square(size=Point(square_size, square_size), origin=Point(0., 0.))
    x_square = np.arange(
        np.min(hexagon_centers), np.max(hexagon_centers), square_size)
    y_square = np.arange(
        np.min(hexagon_centers),
        np.max(hexagon_centers), square_size) + hexagon_size * np.sqrt(3) / 2.

    square_centers = np.array(list(product(x_square, y_square)))
    squares = []
    for square_center in square_centers:
        squares += square_corners(square, Point(square_center[0], square_center[1]))
    squares = np.array(squares)
    square_array = squares.reshape(len(x_square)**2, 4, 2)

    return square_centers, square_array


def plot_fake_hex_image(hexagon_centers, flat_image):
    """Make scatter plot of intensity using hexagon centers.

    Parameters
    ----------
    hexagon_centers : array
        2D array containing center coordinates for each hexagon.
    flat_image : array
        Flattened image corresponding to positions.

    """
    from astropy.visualization import ImageNormalize, ZScaleInterval
    norm_counts = ImageNormalize(flat_image, interval=ZScaleInterval())
    plt.scatter(
        hexagon_centers[:, 0], hexagon_centers[:, 1], c=flat_image, norm=norm_counts)
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.show()


def plot_hex_grid(hexagon_arr, square_arr=None, color=None):
    """Visualize hexgrid.

    Parameters
    ----------
    hexagon_arr : arr
        Array containing coordinates of corners for each hexagon.
    color : str
        Line color.

    """
    if hexagon_arr is not None:
        for hexagon in hexagon_arr:
            plt.plot(hexagon[:, 0], hexagon[:, 1], color=color)
    if square_arr is not None:
        for square in square_arr:
            plt.plot(square[:, 0], square[:, 1], color=color)


def contribution_by_hexagons(
        square_center, square_corners, hexagon_centers, hexagon_arr, dmax):
    """For a pixel, identifies nearby hexagons and computes overlap area.

    Parameters
    ----------
    square_center : tuple
        Center coordinates of pixel.
    square_corners : tuple
        Corner coordinates of pixel.
    hexagon_centers : array
        2D array containing center coordinates for each hexagon.
    hexagon_arr : array
        Array containing coordinates of corners for each hexagon.
    dmax : float
        max(hexagon_size, square_size).

    Returns
    -------
    dictionary
        Contains indices of hexagons with overlap and respective area.

    """
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


def make_mapping(image_size=201, hexagon_size=1. / np.sqrt(3),
                 square_size=1. / np.sqrt(3), layout='pointy',
                 outputname=None):
    """Make calibration input for resampling routine. Compute
       overlap of relevant hexagons with all pixels.

    Parameters
    ----------
    image_size : int
        Number of lenslets in one dimension.
    hexagon_size : float
        Side length of hexagon.
    square_size : float
        Side length of pixel.
    layout : str
        'pointy' or 'square' hexagon layout.
    outputname : str
        Name of output calibration file.

    Returns
    -------
    list
        List of dictionaries containing hexagon index and overlap
        area for each pixel.

    """

    hexagon_centers = make_hex_coordinates(image_size, hexagon_size=hexagon_size)
    hexagon_arr = make_hexagon_array(image_size, hexagon_size, layout)

    square_centers, square_arr = make_pixel_overlay_grid(
        image_size, hexagon_size=hexagon_size,
        square_size=square_size, layout=layout)

    dmax = max(hexagon_size, square_size)
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


def resample_image_cube(
        image_cube, clip_infos, hexagon_size=1 / np.sqrt(3)):
    """Resample hexagonal image cube onto a rectilinear grid.

    Parameters
    ----------
    image_cube : array
        Hexagonal image cube (wave * y_index * x_index).
    clip_infos : list
        List of dictionaries containing hexagon index and overlap
        area for each pixel.
    hexagon_size : float
        Side length of hexagon.

    Returns
    -------
    array
        Image cube on pixel grid.

    """
    flat_cube = flatten_cube(image_cube)
    hexagon_area = hexagon_size**2 * 3 / 2 * np.sqrt(3)
    image_cube = np.zeros([len(flat_cube), len(clip_infos)])
    number_of_pixels = int(np.sqrt(len(clip_infos)))

    for index, clip_info in enumerate(clip_infos):
        if len(clip_info['areas']) > 0:
            image_cube[:, index] = np.sum(
                flat_cube[:, clip_info['hex_indices']] *
                np.array(clip_info['areas']) / hexagon_area, axis=1)

    image_cube = image_cube.reshape(
        len(image_cube), number_of_pixels, number_of_pixels)
    image_cube = np.swapaxes(image_cube, 1, 2)
    image_cube = image_cube[:, 18:-68, 46:-40]
    return image_cube


def resample_image_cube_file(filename, clip_info_file, hexagon_size=1 / np.sqrt(3)):
    clip_infos = pickle.load(open(clip_info_file, "rb"))

    image_cube = fits.getdata(filename)
    header = fits.getheader(filename)

    image_cube = resample_image_cube(
        image_cube, clip_infos, hexagon_size=1 / np.sqrt(3))

    outputfilename = os.path.splitext(filename)[0] + '_resampled.fits'
    fits.writeto(outputfilename, image_cube, header, overwrite=True)


def distance_from_points(point, points):
    distance = np.sqrt((point[0] - points[:, 0])**2 + (point[1] - points[:, 1])**2)
    return distance


def find_neighbours(point, points, hexagon_size=1 / np.sqrt(3), include_center=False):
    distance_to_neighbour = hexagon_size * np.sqrt(3) + 1e-8
    distance = distance_from_points(point, points)
    if include_center:
        neighbour_mask = distance < distance_to_neighbour
    else:
        neighbour_mask = np.logical_and(distance > 0, distance < distance_to_neighbour)
    return neighbour_mask


def find_neighbour_indices(hexagon_centers, hexagon_size=1. / np.sqrt(3), include_center=False):
    neighbours = []
    for hexagon in tqdm(hexagon_centers):
        neighbours.append(np.where(find_neighbours(
            hexagon, hexagon_centers, hexagon_size, include_center))[0])
    return neighbours


def make_calibration_neighbour_indices(
        image_size, hexagon_size=1. / np.sqrt(3),
        include_center=False, layout='pointy',
        outputname=None):

    hexagon_centers = make_hex_coordinates(
        image_size, hexagon_size, layout)

    neighbour_indices = find_neighbour_indices(
        hexagon_centers, hexagon_size, include_center)

    if outputname is not None:
        with open(outputname, 'w') as fout:
            json.dump(neighbour_indices, fout, cls=NumpyEncoder)

    return neighbour_indices


def median_filter_hex_image(flat_image, index_list):
    smooth_image = np.zeros_like(flat_image)
    for pix_index, value in enumerate(flat_image):
        neighbour_mask = index_list[pix_index]
        median_value = np.median(flat_image[neighbour_mask])
        smooth_image[pix_index] = median_value
    return smooth_image


def median_filter_hex_cube(flat_cube, index_list):
    smooth_cube = np.zeros_like(flat_cube)

    for pix_index, neighbour_mask in enumerate(index_list):
        smooth_cube[:, pix_index] = np.median(flat_cube[:, neighbour_mask], axis=1)
    return smooth_cube


def mad_std_hex_cube(flat_cube, index_list):
    surrounding_robust_std_dev = np.zeros_like(flat_cube)

    for pix_index, neighbour_mask in enumerate(index_list):
        surrounding_robust_std_dev[:, pix_index] = mad_std(flat_cube[:, neighbour_mask], axis=1)
    return surrounding_robust_std_dev


def mask_outliers_from_absolute_deviation(flat_cube, sigma=12):
    good_pixel = flat_cube > 0.
    for wave_idx in range(len(flat_cube)):
        values = flat_cube[wave_idx][good_pixel[wave_idx]]
        clipped = sigma_clip(sigma_clip(values, sigma=sigma, cenfunc=np.median, stdfunc=mad_std))
        good_pixel[wave_idx][good_pixel[wave_idx]] = ~clipped.mask
    return ~good_pixel


def bad_corrected_smoothed_cube(flat_cube, bad_pixel_map, index_list):
    smoothed_cube = median_filter_hex_cube(flat_cube, index_list)
    flat_cube[bad_pixel_map] = smoothed_cube[bad_pixel_map]
    corrected_smooth_cube = median_filter_hex_cube(flat_cube, index_list)
    return corrected_smooth_cube


def detect_outliers_median_filter(flat_ivar, index_list, sigma=12):
    bad_pixel_map = mask_outliers_from_absolute_deviation(flat_ivar, sigma=sigma)
    corrected_smooth_cube = bad_corrected_smoothed_cube(
        flat_ivar.copy(), bad_pixel_map, index_list)

    median_filter_mask = (1e-20 + flat_ivar) / (1e-20 + corrected_smooth_cube) > 1.1
    return median_filter_mask


def fix_bad_pixel(data, ivar, index_list, sigma=12):
    flat_data = flatten_cube(data)
    flat_ivar = flatten_cube(ivar)

    median_filter_mask = detect_outliers_median_filter(flat_ivar, index_list, sigma=sigma)
    flat_ivar[median_filter_mask] = 0.

    smooth_data = bad_corrected_smoothed_cube(flat_data, median_filter_mask, index_list)
    flat_data[median_filter_mask] = smooth_data[median_filter_mask]

    data = deflatten_cube(flat_data)
    ivar = deflatten_cube(flat_ivar)
    return data, ivar, deflatten_cube(median_filter_mask)


# with open('neighbour_indices.json') as json_data:
#     index_list = json.load(json_data)
# hexagon_centers = make_hex_coordinates(201)
#
# image_cube = fits.getdata(
#     '/home/samland/science/sphere_calibration/test_data/ifs_test_data/YJ/science_cubes/waffle_cubes/SPHER.2015-05-14T06_40_52.356IFS_STAR_CENTER_WAFFLE_RAW_cube.fits')
# variance = fits.getdata(
#     '/home/samland/science/sphere_calibration/test_data/ifs_test_data/YJ/science_cubes/waffle_cubes/SPHER.2015-05-14T06_40_52.356IFS_STAR_CENTER_WAFFLE_RAW_cube.fits', 2)
# flat_cube = flatten_cube(image_cube)
# flat_ivar = flatten_cube(variance)
# # from astropy.visualization import ZScaleInterval, ImageNormalize
# #
# #
# # data1, ivar1, bad = fix_bad_pixel(image_cube, variance, index_list)
# #
# data2, ivar2, mask2 = smoothandmask(image_cube, variance, index_list)
# # data3, ivar3, mask3 = smoothandmask2(image_cube, variance, index_list, sigma=12)
# #
# fits.writeto('data_simple.fits', data2, overwrite=True)
# fits.writeto('ivar_simple.fits', ivar2, overwrite=True)
# fits.writeto('data_abs.fits', data3, overwrite=True)
# fits.writeto('ivar_abs.fits', ivar3, overwrite=True)

# fits.writeto('1st_iter.fits', deflatten_cube(flat_variance), overwrite=True)
#
# flat_variance[zero_mask] = smoothed_variance[zero_mask]
#
# smoothed_variance = median_filter_hex_cube(flat_variance, index_list)
#
# bad_mask = flat_variance_orig < smoothed_variance / 10.
#
# flat_variance_orig[bad_mask] = 0.
#
#
# fits.writeto('deviation.fits', deflatten_cube(deviation))
