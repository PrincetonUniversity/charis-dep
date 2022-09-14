#!/usr/bin/env python

import sys

import numpy as np
from astropy.io import fits
from bokeh.io import curdoc, output_file  # , show
from bokeh.layouts import column, gridplot, row, widgetbox
from bokeh.models import LinearColorMapper  # , LogTicker, ColorBar
from bokeh.models import ColumnDataSource, HoverTool, LayoutDOM, Range1d, LinearAxis
from bokeh.models.widgets import Panel, Slider, Tabs
from bokeh.plotting import figure, show
from bokeh.transform import linear_cmap, log_cmap
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


# output_file("hexplot.html")
TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"
normalize = True

# filename = '/home/samland/science/charis_paper/2017/optext/CENTER/SPHER.2017-09-28T07:51:32.456_cube_DIT_000.fits'
filename = sys.argv[1]
hdu_list = fits.open(filename)
# image_cube = hdu_list[1].data
image_cube = fits.getdata(filename)
image_cube[~np.isfinite(image_cube)] = 0.

if image_cube.ndim == 2:
    image_cube = np.expand_dims(image_cube, axis=0)
N = image_cube.shape[0]

# image_cube_drh = fits.getdata(file_drh)
X, Y, image_cube = crop_hex_cube(image_cube, i1=16, i2=-16, j1=16, j2=-16)
centers = np.vstack([X.ravel(), Y.ravel()]).T

# image_cube2 = fits.getdata(filename2)
# _, _, image_cube2 = crop_hex_cube(image_cube2, i1=16, i2=-16, j1=16, j2=-16)
# image_cube2 = flatten_cube(image_cube2)

image_cube = flatten_cube(image_cube)

mask = image_cube[0] == 0.
centers = centers[~mask]
image_cube = image_cube[:, ~mask]

try:
    variance_cube = hdu_list[2].data
    if variance_cube.ndim == 2:
        variance_cube = np.expand_dims(variance_cube, axis=0)
    _, _, variance_cube = crop_hex_cube(variance_cube, i1=16, i2=-16, j1=16, j2=-16)
    variance_cube = flatten_cube(variance_cube)
    variance_cube = variance_cube[:, ~mask]
    # norm_variance = ImageNormalize(variance_cube[3], interval=ZScaleInterval())
except:
    variance_cube = None
hdu_list.close()


# image_cube2 = image_cube2[:, ~mask]

q, r = cartesian_to_axial(
    centers[:, 0], centers[:, 1], size=1 / np.sqrt(3), orientation="pointytop", aspect_scale=1)
# df = pd.DataFrame(
# data=dict(q=q, r=r, counts=image_slice.flatten(), x=centers[:, 0], y=centers[:, 1]))

if normalize:
    norm_factor = np.max(image_cube[0, :])
else:
    norm_factor = 1.

if variance_cube is None:
    source = ColumnDataSource(data=dict(q=q, r=r, counts=image_cube[0, :] / norm_factor,
                                        variance=np.zeros_like(image_cube[0, :]), x=centers[:, 0], y=centers[:, 1]))
else:
    source = ColumnDataSource(data=dict(q=q, r=r, counts=image_cube[0, :] / norm_factor,
                                        variance=variance_cube[0, :], x=centers[:, 0], y=centers[:, 1]))
# source2 = ColumnDataSource(data=dict(q=q, r=r, counts=image_cube2[0, :],
#                                      variance=variance_cube[0, :], x=centers[:, 0], y=centers[:, 1]))

left = figure(tools=TOOLS, match_aspect=True,
              background_fill_color='#ffffff',
              x_range=Range1d(172 - 110, 172 + 110),
              y_range=Range1d(146 - 110, 146 + 110))

# left.add_layout(LinearAxis(), 'above')
# left.add_layout(LinearAxis(), 'right')
# left.xaxis.axis_label_text_font_size = "25pt"
left.xaxis.major_label_text_font_size = "25pt"
# left.yaxis.axis_label_text_font_size = "25pt"
left.yaxis.major_label_text_font_size = "25pt"
left.grid.visible = False
# left.sizing_mode = 'scale_width'

# right = figure(tools=TOOLS, title="SPHERE IFS Variance", match_aspect=True,
#                x_range=left.x_range, y_range=left.y_range, background_fill_color='#440154')
# right = figure(tools=TOOLS, title="SPHERE DRH", match_aspect=True,
#                background_fill_color='#440154')

# right.grid.visible = False
# right.sizing_mode = 'scale_width'

norm_counts = ImageNormalize(image_cube[0], interval=ZScaleInterval())
# norm_counts2 = ImageNormalize(image_cube2[0], interval=ZScaleInterval())

# mask_drh = image_cube_drh[10] != 0.
# norm_drh = ImageNormalize(image_cube_drh[10][mask_drh], interval=ZScaleInterval())
# cmap = 'Inferno256'
# cmap = 'Viridis256'
cmap = palettes.Viridis256[::-1]

a = left.hex_tile(
    q='q', r='r', line_color=None, source=source,
    fill_color=linear_cmap('counts', cmap, norm_counts.vmin, norm_counts.vmax),
    # fill_color=log_cmap('counts', cmap, 0.005, 1),
    hover_color="pink", hover_alpha=0.8)

# b = right.hex_tile(
#     q='q', r='r', line_color=None, source=source,
#     fill_color=linear_cmap('variance', 'Inferno256', norm_variance.vmin, norm_variance.vmax),
#     hover_color="pink", hover_alpha=0.8)

# b = right.hex_tile(
#     q='q', r='r', line_color=None, source=source2,
#     fill_color=linear_cmap('counts', 'Inferno256', norm_counts2.vmin, norm_counts2.vmax),
#     hover_color="pink", hover_alpha=0.8)

# b = right.image(image=[image_cube_drh[10]], x=0, y=0, dw=291, dh=291,
#                 color_mapper=LinearColorMapper(palette='Inferno256', low=norm_drh.vmin, high=norm_drh.vmax))

if N > 1:
    slider = Slider(start=0, end=(N - 1), value=0, step=1, title="Wavelength")
    # slider2 = Slider(start=0, end=(N - 1), value=0, step=1, title="Wavelength")

    def update(attr, old, new):
        if normalize:
            norm_factor = np.max(image_cube[int(slider.value), :])
        else:
            norm_factor = 1.

        source.data['counts'] = image_cube[int(
            slider.value), :] / norm_factor
        norm_counts = ImageNormalize(image_cube[int(slider.value)], interval=ZScaleInterval())

        # source.data['variance'] = variance_cube[int(slider.value), :]
        # norm_variance = ImageNormalize(variance_cube[slider_value], interval=ZScaleInterval())

        # source2.data['counts'] = image_cube2[int(slider.value), :]
        # norm_counts2 = ImageNormalize(image_cube2[slider_value], interval=ZScaleInterval())

        a.glyph.fill_color = linear_cmap('counts', cmap, norm_counts.vmin, norm_counts.vmax)
        # a.glyph.fill_color = log_cmap('counts', cmap, 0.005, 1)
        # b.glyph.fill_color = linear_cmap('variance', 'Inferno256', norm_variance.vmin, norm_variance.vmax)
        # b.glyph.fill_color = linear_cmap('counts', 'Inferno256', norm_counts2.vmin, norm_counts2.vmax)

    # def update2(attr, old, new):
    #
    #     source2.data['counts'] = image_cube2[int(slider.value), :]
    #     norm_counts2 = ImageNormalize(image_cube2[slider_value], interval=ZScaleInterval())
    #
    #     # source.data['variance'] = variance_cube[int(slider.value), :]
    #     # norm_variance = ImageNormalize(variance_cube[slider_value], interval=ZScaleInterval())
    #
    #     # a.glyph.fill_color = linear_cmap('counts', 'Inferno256', norm_counts.vmin, norm_counts.vmax)
    #     # b.glyph.fill_color = linear_cmap('variance', 'Inferno256', norm_variance.vmin, norm_variance.vmax)
    #     b.glyph.fill_color = linear_cmap('counts', 'Inferno256', norm_counts2.vmin, norm_counts2.vmax)

    slider.on_change('value', update)
    # slider2.on_change('value', update2)

hover = HoverTool(tooltips=[
    ("count", "@counts"),
    ("variance", "@variance"),
    # ("(q,r)", "(@q, @r)"),
    ("(x,y)", "(@x, @y)")],
    mode="mouse", point_policy="follow_mouse", renderers=[a])  # , b])

left.add_tools(hover)
# right.add_tools(hover)

# p = gridplot([[left, right],
#               [slider, slider2]], plot_width=750, plot_height=750)
if N > 1:
    p = gridplot([[left, None],
                  [slider, None]], width=750, height=750)
else:
    p = left

# p = layout(t([left, slider])
# p.sizing_mode = 'scale_width'
# p.show()
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
