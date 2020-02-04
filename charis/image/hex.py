from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np


def cartesian_to_axial(x, y, size, orientation, aspect_scale=1):
    ''' Map Cartesion *(x,y)* points to axial *(q,r)* coordinates of enclosing
    tiles.

    This function is adapted to overwrite basic Bokeh coordinate
    rounding behavior. For original function see Bokeh.
    Original implementation based on:

        https://www.redblobgames.com/grids/hexagons/#pixel-to-hex

    Args:
        x (array[float]) :
            A NumPy array of x-coordinates to convert

        y (array[float]) :
            A NumPy array of y-coordinates to convert

        size (float) :
            The size of the hexagonal tiling.

            The size is defined as the distance from the center of a hexagon
            to the top corner for "pointytop" orientation, or from the center
            to a side corner for "flattop" orientation.

        orientation (str) :
            Whether the hex tile orientation should be "pointytop" or
            "flattop".

        aspect_scale (float, optional) :
            Scale the hexagons in the "cross" dimension.

            For "pointytop" orientations, hexagons are scaled in the horizontal
            direction. For "flattop", they are scaled in vertical direction.

            When working with a plot with ``aspect_scale != 1``, it may be
            useful to set this value to match the plot.

    Returns:
        (array[int], array[int])

    '''
    HEX_FLAT = [2.0 / 3.0, 0.0, -1.0 / 3.0, np.sqrt(3.0) / 3.0]
    HEX_POINTY = [np.sqrt(3.0) / 3.0, -1.0 / 3.0, 0.0, 2.0 / 3.0]

    coords = HEX_FLAT if orientation == 'flattop' else HEX_POINTY

    x = x / size * (aspect_scale if orientation == "pointytop" else 1)
    y = -y / size / (aspect_scale if orientation == "flattop" else 1)

    q = coords[0] * x + coords[1] * y
    r = coords[2] * x + coords[3] * y

    return q, r
