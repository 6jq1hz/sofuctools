import sys
import os.path
import numpy as np
from skimage import io
from skimage.color import rgb2hsv, rgba2rgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    img = io.imread(sys.argv[1])
    fname = os.path.basename(sys.argv[1])
else:
    quit()

if img.shape[2] == 4:
    img = rgba2rgb(img)

img_sv = rgb2hsv(img)[:, :, 1:3]
sv_data = np.reshape(img_sv, (img_sv.shape[0]*img_sv.shape[1],2))

fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(8, 4), num=fname+"_hist")
ax1.imshow(img)
ax1.set_axis_off()
ax2.set_aspect(1, adjustable='box')
hist = ax2.hist2d(sv_data[:,0],sv_data[:,1],
    bins=64, range=[[0,1],[0,1]], norm=mcolors.PowerNorm(0.6),
    cmax=img_sv.shape[0]*img_sv.shape[1]*0.1)
plt.ylim(0,1)
plt.xlim(0,1)
plt.xlabel("saturation")
plt.ylabel("value")

plt.show()
