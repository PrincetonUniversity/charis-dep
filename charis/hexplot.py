#!/usr/bin/env python

import sys

import numpy as np
from astropy.io import fits
from bokeh.io import curdoc
from bokeh.layouts import gridplot
# from bokeh.models import LinearColorMapper  # , LogTicker, ColorBar
from bokeh.models import ColumnDataSource, HoverTool, Range1d, Slider
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh import palettes
from bokeh.util.hex import axial_to_cartesian, hexbin

from charis.image.hex import cartesian_to_axial

import pandas as pd
from astropy.visualization import (AsymmetricPercentileInterval,
                                   ImageNormalize, LinearStretch, LogStretch,
                                   MinMaxInterval, PercentileInterval,
                                   ZScaleInterval)


def crop_hex_cube(image_cube, i1=None, i2=None, j1=None, j2=None):
    image_size = image_cube.shape[-1]
    # x = np.arange(-1 * (image_size // 2), (image_size // 2) + 1) * 1.
    x = np.arange(0, image_size, dtype='float64')
    x, y = np.meshgrid(x, x)
    x[::2] -= (0.5 + 1e-8)
    y *= np.sqrt(3) / 2

    return x[i1:i2, j1:j2], y[i1:i2, j1:j2], image_cube[:, i1:i2, j1:j2]


def prepare_slice(x, y, image_slice):

    x, y, cells, image_data = set_points_cells(x, y, image_slice)

    indx = np.where(image_data != 0)
    cells = np.asarray(cells)[indx]
    image_data = image_data[indx]

    return x, y, cells, image_data


def plot_slice(x, y, cells, image_data):
    norm = ImageNormalize(image_data, interval=ZScaleInterval())
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.tripcolor(x[::-1], y, cells, facecolors=image_data,
                  cmap='inferno', norm=norm)  # cmap='cubehelix_r')
    plt.show()


def flatten_cube(image_cube):
    number_of_frames = image_cube.shape[0]
    number_of_pixels = image_cube.shape[-1]**2
    return image_cube.reshape(number_of_frames, number_of_pixels)


def create_hexplot(filename: str) -> None:
    """Create and serve a hexplot visualization for the given FITS file.
    
    Args:
        filename: Path to the FITS file to visualize
    """
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"
    normalize = False

    hdu_list = fits.open(filename)
    image_cube = fits.getdata(filename)
    image_cube[~np.isfinite(image_cube)] = 0.

    if image_cube.ndim == 2:
        image_cube = np.expand_dims(image_cube, axis=0)
    N = image_cube.shape[0]

    X, Y, image_cube = crop_hex_cube(image_cube, i1=16, i2=-16, j1=16, j2=-16)
    centers = np.vstack([X.ravel(), Y.ravel()]).T
    image_cube = flatten_cube(image_cube)

    mask = image_cube[0] == 0.
    centers = centers[~mask]
    image_cube = image_cube[:, ~mask]

    try:
        inv_variance_cube = hdu_list[2].data
        if inv_variance_cube.ndim == 2:
            inv_variance_cube = np.expand_dims(inv_variance_cube, axis=0)
        _, _, inv_variance_cube = crop_hex_cube(inv_variance_cube, i1=16, i2=-16, j1=16, j2=-16)
        inv_variance_cube = flatten_cube(inv_variance_cube)
        inv_variance_cube = inv_variance_cube[:, ~mask]
    except:
        inv_variance_cube = None
    hdu_list.close()

    q, r = cartesian_to_axial(
        centers[:, 0], centers[:, 1], size=1 / np.sqrt(3), orientation="pointytop", aspect_scale=1)

    if normalize:
        norm_factor = np.max(image_cube[0, :])
    else:
        norm_factor = 1.

    if inv_variance_cube is None:
        source = ColumnDataSource(data=dict(q=q, r=r, counts=image_cube[0, :] / norm_factor,
                                          inv_variance=np.zeros_like(image_cube[0, :]), x=centers[:, 0], y=centers[:, 1]))
    else:
        source = ColumnDataSource(data=dict(q=q, r=r, counts=image_cube[0, :] / norm_factor,
                                          inv_variance=inv_variance_cube[0, :], x=centers[:, 0], y=centers[:, 1]))

    left = figure(tools=TOOLS, match_aspect=True,
                 background_fill_color='#ffffff',
                 active_scroll='wheel_zoom')

    left.xaxis.major_label_text_font_size = "25pt"
    left.yaxis.major_label_text_font_size = "25pt"
    left.grid.visible = False

    norm_counts = ImageNormalize(image_cube[0], interval=ZScaleInterval())
    cmap = palettes.Viridis256[::-1]

    a = left.hex_tile(
        q='q', r='r', line_color=None, source=source,
        fill_color=linear_cmap('counts', cmap, norm_counts.vmin, norm_counts.vmax),
        hover_color="pink", hover_alpha=0.8)

    if N > 1:
        slider = Slider(start=0, end=(N - 1), value=0, step=1, title="Wavelength")

        def update(attr, old, new):
            if normalize:
                norm_factor = np.max(image_cube[int(slider.value), :])
            else:
                norm_factor = 1.

            source.data['counts'] = image_cube[int(
                slider.value), :] / norm_factor
            norm_counts = ImageNormalize(image_cube[int(slider.value)], interval=ZScaleInterval())

            a.glyph.fill_color = linear_cmap('counts', cmap, norm_counts.vmin, norm_counts.vmax)

        slider.on_change('value', update)

    hover = HoverTool(tooltips=[
        ("count", "@counts"),
        ("inverse variance", "@inv_variance"),
        ("(x,y)", "(@x, @y)")],
        mode="mouse", point_policy="follow_mouse", renderers=[a])

    left.add_tools(hover)

    if N > 1:
        p = gridplot([[left, None],
                      [slider, None]], width=750, height=750)
    else:
        p = left

    curdoc().add_root(p)

'''
output_file("slider.html")

p1 = figure(plot_width=300, plot_height=300)
p1.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)
tab1 = Panel(child=p1, title="circle")

p2 = figure(plot_width=300, plot_height=300)
p2.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=3, color="navy", alpha=0.5)
tab2 = Panel(child=p2, title="line")

tabs = Tabs(tabs=[ tab1, tab2 ])

show(tabs)
'''

# show(p)
"""

import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure

from bokeh.settings import settings
N = 100
output_file("test.html", mode='inline')

settings.log_level('trace')
x_ = np.linspace(0, 10, 200)
y_ = np.linspace(0, 10, 200)
z_ = np.linspace(0, 10, N)

x, y, z = np.meshgrid(x_, y_, z_, indexing='xy')

data = np.sin(x + z) * np.cos(y)

source = ColumnDataSource(data=dict(image=[data[:, :, 0]]))

p = figure(x_range=(0, 10), y_range=(0, 10))
p.image(image='image', x=0, y=0, dw=10, dh=10, source=source, palette="Spectral11")

slider = Slider(start=0, end=(N - 1), value=0, step=1, title="Frame")


def update(attr, old, new):
    source.data = dict(image=[data[:, :, slider.value]])


slider.on_change('value', update)
# p2 = gridplot([[p, slider]])
# show(p2)
# curdoc().add_root(column(p, slider))

bokeh serve --show hexplot.py
# show(p2)
"""
