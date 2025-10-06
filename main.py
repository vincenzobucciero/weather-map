from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
from metpy.calc import find_peaks
from haversine import haversine, Unit
import time


start_time = time.time()

f = Dataset(r'data/opendap/wrf5/d01/archive/2025/09/13/wrf5_d01_20250913Z1200.nc', 'r')
lats = f['latitude'][:]
lons = f['longitude'][:]
Lons, Lats = np.meshgrid(lons, lats)

clf = f['CLDFRA_TOTAL'][0, : , :]
slp = f['SLP'][0, : , :]
# Here, we go from transparent white (1,1,1,0) to opaque blue (0,0,1,1).
clf_colors = [(1, 1, 1, 0), (1, 1, 1, 1)]
clf_cmap: LinearSegmentedColormap = LinearSegmentedColormap.from_list("transparent_clouds", clf_colors, N=256)

slp_min_val = 960
slp_max_val = 1100
slp_step = 2
levels = np.arange(slp_min_val, slp_max_val + slp_step, slp_step) # +step to include max_val if it's a multiple of step

m = Basemap(projection='merc',
            llcrnrlat=lats[0], urcrnrlat=lats[-1],
            llcrnrlon=lons[0], urcrnrlon=lons[-1],
            lat_ts=20,
            resolution='h')
m.bluemarble(scale=4)
m.drawcoastlines()
x, y = m(Lons, Lats)
plt.contourf(x, y, clf, cmap=clf_cmap)

plt.contour(x, y,gaussian_filter(slp,sigma=3), colors='yellow', levels=levels, linewidths=.5, linestyles='solid' )


h_y, h_x = find_peaks(slp)
l_y, l_x = find_peaks(slp, maxima=False)

h_lons, h_lats = m(lons[h_x], lats[h_y])
l_lons, l_lats = m(lons[l_x], lats[l_y])

print(len(l_y), len(h_x))

M = np.zeros((len(l_x), len(h_x)), dtype=float)
for l in range(len(l_x)):
    l_value = slp[l_y[l]][l_x[l]]
    for h in range(len(h_x)):
        M[l,h] = abs(l_value-slp[h_y[h]][h_x[h]])/(10*haversine((lats[l_y[l]], lons[l_x[l]]), (lats[h_y[h]], lons[h_x[h]])))

print(M)

threshold = .75e-03

for l in range(len(l_x)):
    value = slp[l_y[l]][l_x[l]]
    print("L",value)
    if value < 1013:
        draw = False
        for h in range(len(h_x)):
            if M[l,h] > threshold:
                draw = True
                break

        if draw:
            label = 'L\n' + str(round(value))
            plt.text(l_lons[l], l_lats[l], label, fontsize=14, color='red', ha='center', va='center', fontweight='bold')

for h in range(len(h_x)):
    value = slp[h_y[h]][h_x[h]]
    print("H", value)
    if value >= 1013:
        draw = False
        for l in range(len(l_x)):
            if M[l,h] > threshold:
                draw = True
                break


        if draw:
            label = 'H\n' + str(round(value))
            plt.text(h_lons[h], h_lats[h], label, fontsize=14, color='blue', ha='center', va='center', fontweight='bold')

plt.savefig("out.png", dpi=1200)

print("Execution time: %s seconds" % (time.time() - start_time))


plt.show()