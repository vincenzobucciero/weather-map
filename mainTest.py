from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter
from haversine import haversine, Unit
import time

# -------------------- Config --------------------
NC_PATH = r"data/opendap/wrf5/d01/archive/2025/09/13/wrf5_d01_20250913Z1200.nc"

# smoothing campo SLP (maggiore = meno punti, più pulito)
GAUSS_SIGMA = 5
# finestra per min/max locali (dispari, maggiore = meno punti)
LOCAL_WIN = 15
# percentile per selezionare aree a gradiente basso (più basso = più selettivo)
GRAD_PCTL = 2
# distanza minima tra centri selezionati (km)
MIN_SEP_KM = 320
# soglia tua per la matrice M
M_THRESHOLD = 0.75e-03

# livelli isobare
SLP_MIN, SLP_MAX, SLP_STEP = 960, 1100, 2

# colori badge
COLOR_H_FACE = "#1976D2"
COLOR_H_EDGE = "#0B3D91"
COLOR_L_FACE = "#D32F2F"
COLOR_L_EDGE = "#8B0000"


# -------------------- Utils --------------------
def put_badge(ax, x, y, symbol, value, facecolor, edgecolor):
    """Disegna un badge circolare con lettera e valore sotto."""
    ax.text(
        x, y, symbol,
        fontsize=12, color="white", weight="bold",
        ha="center", va="center",
        bbox=dict(boxstyle="circle,pad=0.18",
                  facecolor=facecolor, edgecolor=edgecolor,
                  linewidth=1.0, alpha=0.95),
        zorder=7,
        path_effects=[pe.withStroke(linewidth=1.5, foreground="black", alpha=0.35)]
    )
    ax.annotate(
        f"{int(round(float(value)))}",
        xy=(x, y), xycoords="data",
        xytext=(0, -10), textcoords="offset points",
        ha="center", va="top",
        fontsize=7, color=facecolor, weight="bold",
        zorder=7,
        path_effects=[pe.withStroke(linewidth=1.8, foreground="white", alpha=0.9)]
    )


def filter_by_distance(y_idx, x_idx, lats, lons, min_km=300):
    """Greedy: tiene il primo punto e scarta i successivi più vicini di min_km."""
    keep_y, keep_x = [], []
    for i in range(len(y_idx)):
        lat_i, lon_i = lats[y_idx[i]], lons[x_idx[i]]
        if all(haversine((lat_i, lon_i), (lats[yy], lons[xx])) > min_km
               for yy, xx in zip(keep_y, keep_x)):
            keep_y.append(y_idx[i])
            keep_x.append(x_idx[i])
    return np.array(keep_y), np.array(keep_x)


# -------------------- Main --------------------
start_time = time.time()

# Caricamento dati
f = Dataset(NC_PATH, "r")
lats = f["latitude"][:]
lons = f["longitude"][:]
Lons, Lats = np.meshgrid(lons, lats)

clf = f["CLDFRA_TOTAL"][0, :, :]
slp_raw = f["SLP"][0, :, :]

# Smoothing per stabilizzare gradiente ed estremi
slp = gaussian_filter(slp_raw, sigma=GAUSS_SIGMA)

# Colormap per nubi trasparente -> bianco
clf_cmap = LinearSegmentedColormap.from_list(
    "transparent_clouds", [(1, 1, 1, 0), (1, 1, 1, 1)], N=256
)

# Livelli isobare
levels = np.arange(SLP_MIN, SLP_MAX + SLP_STEP, SLP_STEP)

# --- Mappa ---
m = Basemap(projection="merc",
            llcrnrlat=lats[0], urcrnrlat=lats[-1],
            llcrnrlon=lons[0], urcrnrlon=lons[-1],
            lat_ts=20, resolution="h")
m.bluemarble(scale=4)
m.drawcoastlines()

x, y = m(Lons, Lats)
plt.contourf(x, y, clf, cmap=clf_cmap, zorder=1)
plt.contour(x, y, slp, colors="yellow", levels=levels,
            linewidths=.5, linestyles="solid", zorder=2)

# === Gradiente & Laplaciano ===
gy, gx = np.gradient(slp)           # dSLP/dlat, dSLP/dlon (in grid units)
grad_mag = np.hypot(gx, gy)

gy_y, _ = np.gradient(gy)
_, gx_x = np.gradient(gx)
lap = gx_x + gy_y                   # Laplaciano

# === Candidati: gradiente basso ===
tau = np.percentile(grad_mag, GRAD_PCTL)
crit_mask = grad_mag < tau

# Estremi locali su finestra
is_local_min = (slp == minimum_filter(slp, size=LOCAL_WIN)) & crit_mask
is_local_max = (slp == maximum_filter(slp, size=LOCAL_WIN)) & crit_mask

# Classificazione con Laplaciano:
#   lap < 0 -> massimo (H),  lap > 0 -> minimo (L)
H_mask = is_local_max & (lap < 0)
L_mask = is_local_min & (lap > 0)

# Indici [y, x]
h_y, h_x = np.where(H_mask)
l_y, l_x = np.where(L_mask)

# Distanza minima tra centri per pulizia
h_y, h_x = filter_by_distance(h_y, h_x, lats, lons, min_km=MIN_SEP_KM)
l_y, l_x = filter_by_distance(l_y, l_x, lats, lons, min_km=MIN_SEP_KM)

# (Opzionale) limiti massimi: tieni i più prominenti
MAX_KEEP = 200
if len(h_y) > MAX_KEEP:
    order = np.argsort(slp[h_y, h_x])[::-1]  # massimi più alti
    h_y, h_x = h_y[order[:MAX_KEEP]], h_x[order[:MAX_KEEP]]
if len(l_y) > MAX_KEEP:
    order = np.argsort(slp[l_y, l_x])        # minimi più bassi
    l_y, l_x = l_y[order[:MAX_KEEP]], l_x[order[:MAX_KEEP]]

# Proiezione su mappa
h_lons, h_lats = m(lons[h_x], lats[h_y])
l_lons, l_lats = m(lons[l_x], lats[l_y])

print("N. minimi (L), massimi (H):", len(l_y), len(h_y))

# === Matrice M (tua logica) ===
M = np.zeros((len(l_x), len(h_x)), dtype=float)
for li in range(len(l_x)):
    l_value = slp[l_y[li], l_x[li]]
    for hi in range(len(h_x)):
        dist_km = haversine(
            (lats[l_y[li]], lons[l_x[li]]),
            (lats[h_y[hi]], lons[h_x[hi]]),
            unit=Unit.KILOMETERS
        )
        dist_km = max(dist_km, 1e-6)  # evita zero
        M[li, hi] = abs(l_value - slp[h_y[hi], h_x[hi]]) / (10 * dist_km)

print("M shape:", M.shape)

# === Disegno etichette a badge ===
ax = plt.gca()

for li in range(len(l_x)):
    value = slp[l_y[li], l_x[li]]
    if value < 1013 and np.any(M[li, :] > M_THRESHOLD):
        put_badge(ax, l_lons[li], l_lats[li], "L", value,
                  facecolor=COLOR_L_FACE, edgecolor=COLOR_L_EDGE)

for hi in range(len(h_x)):
    value = slp[h_y[hi], h_x[hi]]
    if value >= 1013 and np.any(M[:, hi] > M_THRESHOLD):
        put_badge(ax, h_lons[hi], h_lats[hi], "H", value,
                  facecolor=COLOR_H_FACE, edgecolor=COLOR_H_EDGE)

plt.savefig("out_badges.png", dpi=1200)
print("Execution time: %s seconds" % (time.time() - start_time))
plt.show()
