from trap.regressor_selection import find_N_unique_samples
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from natsort import natsorted
from trap.plotting_tools import plot_scale

from tqdm import tqdm

directory = '/home/masa4294/science/IFS_flat/archive/'

files = natsorted(glob(directory+"*fits"))

bpm = fits.getdata('/home/masa4294/science/python/projects/charis-dep/charis/calibrations/SPHERE/YH/mask.fits')
bpm = bpm.astype('bool')

images = []
headers = []
times = np.array([1.650726, 4., 10., 25.])
factors = times / times[0]

# yx_pixel = [1183, 1112]


mask = np.ones((2048, 2048)).astype('bool')
border_size = 512
mask[border_size:(2048-border_size), border_size:(2048-border_size)] = True

test = find_N_unique_samples(N=500, yx_dim=(2048, 2048), regressor_pool_mask=mask)

mask = np.logical_and(test, bpm)

pixels = np.argwhere(mask)

for file in files:
    images.append(fits.getdata(os.path.join(directory, file)))
    headers.append(fits.getheader(os.path.join(directory, file)))

images = np.median(np.array(images), axis=1)

# y = mx + c
# c = images[0]
# m = (images[1] - images[0]) / (times[1] - times[0])

prediction_10 = images[1] * 2.5
prediction_25 = images[1] * 6.25
deviation_10 = prediction_10 - images[2]
deviation_25 = prediction_25 - images[3]

# prediction_10 = y(x=10, m=m, c=c)
# prediction_25 = y(x=25, m=m, c=c)
# plt.plot(times, images[:, yx_pixel[0], yx_pixel[1]], 'o')
plt.close()
# for pixel in tqdm(pixels):
# print(pixel[0], pixel[1])
plt.scatter(x=prediction_10[mask], y=images[2][mask], alpha=0.5, s=3, color='black')
plt.scatter(x=prediction_25[mask], y=images[3][mask], alpha=0.5, s=3, color='red')
plt.plot(np.arange(0, 60000), np.arange(0, 60000), '--')
plt.xlabel('Predicted counts')
plt.ylabel('Actual counts')
# images[:, pixel[0], pixel[1]], alpha=0.5, color='black')
plt.show()
