#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
from metpy.calc import find_peaks
from haversine import haversine, Unit
import time

# =========================
# Configurazione
# =========================
NC_PATH = r"data/opendap/wrf5/d02/archive/2025/09/13/wrf5_d02_20250913Z1200.nc"

# Isoipse SLP
SLP_MIN, SLP_MAX, SLP_STEP = 960, 1100, 2

# Smoothing per SLP (consigliato 2–4)
SLP_SIGMA = 4

# Criteri etichette (scegli assoluto o percentili)
USE_PERCENTILES = True
P_LOW, P_HIGH = 10, 90   # usati se USE_PERCENTILES=True
ABS_SPLIT = 1013.0       # usato se USE_PERCENTILES=False

# Metrica L–H: M = |ΔP| / (10 * distanza_km)
M_THRESHOLD = 7.5e-4     # ~0.00075 hPa / (10 km)
NEAR_RADIUS_KM = 700.0   # accoppia L–H solo entro questo raggio

# Estetica testo etichette
LABEL_FONTSIZE = 8

# =========================
# Avvio timer
# =========================
t0 = time.time()

# =========================
# Lettura dati
# =========================
f = Dataset(NC_PATH, "r")
lats = f["latitude"][:]      # (ny,)
lons = f["longitude"][:]     # (nx,)
print(lats)
print(lons)
Lons, Lats = np.meshgrid(lons, lats)  # (ny, nx)

# Variabiliprof 
clf = f["CLDFRA_TOTAL"][0, :, :]     # [0] perché time=1
slp = f["SLP"][0, :, :]

# =========================
# Pre-elaborazione SLP
# =========================
# SLP smussata per isoipse e rilevamento picchi
slp_s = gaussian_filter(slp, sigma=SLP_SIGMA)

# Soglie per H/L
if USE_PERCENTILES:
    p10, p90 = np.percentile(slp, [P_LOW, P_HIGH])
    low_cut, high_cut = p10, p90
else:
    low_cut, high_cut = ABS_SPLIT, ABS_SPLIT

# Livelli per isoipse
levels = np.arange(SLP_MIN, SLP_MAX + SLP_STEP, SLP_STEP)

# =========================
# Rilevamento picchi (su SLP smussata)
# =========================
h_y, h_x = find_peaks(slp_s)               # massimi (H)
l_y, l_x = find_peaks(slp_s, maxima=False) # minimi (L)

print(f"Picchi trovati (prima del filtro M): H={len(h_y)}  L={len(l_y)}")

# =========================
# Basemap (limiti robusti)
# =========================
m = Basemap(
    projection="merc",
    llcrnrlat=float(np.min(lats)), urcrnrlat=float(np.max(lats)),
    llcrnrlon=float(np.min(lons)), urcrnrlon=float(np.max(lons)),
    lat_ts=20, resolution="h"
)
m.bluemarble(scale=4)
m.drawcoastlines(linewidth=0.6)

x, y = m(Lons, Lats)

# =========================
# Layer: nuvolosità + isoipse SLP smussata
# =========================
# Colormap: bianco trasparente -> bianco opaco
clf_cmap = LinearSegmentedColormap.from_list(
    "transparent_clouds",
    [(1, 1, 1, 0), (1, 1, 1, 1)],
    N=256
)
plt.contourf(x, y, clf, levels=12, cmap=clf_cmap)
plt.contour(x, y, slp_s, levels=levels, colors="yellow", linewidths=0.8, linestyles="solid")

# =========================
# Etichette H/L con filtro M e soglie
# =========================
def good_L(val):
    return (val < low_cut) if USE_PERCENTILES else (val < ABS_SPLIT)

def good_H(val):
    return (val >= high_cut) if USE_PERCENTILES else (val >= ABS_SPLIT)

# Prepara coordinate (geografiche) dei picchi
H_lon1d = lons[h_x] if len(h_x) else np.array([])
H_lat1d = lats[h_y] if len(h_y) else np.array([])
L_lon1d = lons[l_x] if len(l_x) else np.array([])
L_lat1d = lats[l_y] if len(l_y) else np.array([])

# Conteggio etichette finali
nL_draw = nH_draw = 0

# --- Disegna L ---
for iL in range(len(l_x)):
    ly, lx = l_y[iL], l_x[iL]
    pL = slp[ly, lx]  # valore reale (non smussato)
    if not good_L(pL):
        continue

    # Verifica accoppiamento con almeno un H entro raggio e M > soglia
    draw = False
    for iH in range(len(h_x)):
        hy, hx = h_y[iH], h_x[iH]
        # distanza geodetica
        d_km = haversine((L_lat1d[iL], L_lon1d[iL]),
                         (H_lat1d[iH], H_lon1d[iH]),
                         unit=Unit.KILOMETERS)
        if d_km == 0.0 or d_km > NEAR_RADIUS_KM:
            continue
        pH = slp[hy, hx]  # valore reale
        M = abs(pL - pH) / (10.0 * d_km)
        if M > M_THRESHOLD:
            draw = True
            break

    if draw:
        # posizioni in mappa
        X, Y = m(L_lon1d[iL], L_lat1d[iL])
        # Estrai valori scalari se sono array/lista
        if isinstance(X, (np.ndarray, list)):
            X = X[0]
        if isinstance(Y, (np.ndarray, list)):
            Y = Y[0]
        plt.text(
            X, Y,
            f"L\n{round(float(pL))}",
            fontsize=LABEL_FONTSIZE, color="red",
            ha="center", va="center", fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.2")
        )
        nL_draw += 1

# --- Disegna H ---
for iH in range(len(h_x)):
    hy, hx = h_y[iH], h_x[iH]
    pH = slp[hy, hx]
    if not good_H(pH):
        continue

    draw = False
    for iL in range(len(l_x)):
        ly, lx = l_y[iL], l_x[iL]
        d_km = haversine((L_lat1d[iL], L_lon1d[iL]),
                         (H_lat1d[iH], H_lon1d[iH]),
                         unit=Unit.KILOMETERS)
        if d_km == 0.0 or d_km > NEAR_RADIUS_KM:
            continue
        pL = slp[ly, lx]
        M = abs(pL - pH) / (10.0 * d_km)
        X, Y = m(H_lon1d[iH], H_lat1d[iH])
        # Estrai valori scalari se sono array/lista
        if isinstance(X, (np.ndarray, list)):
            X = X[0]
        if isinstance(Y, (np.ndarray, list)):
            Y = Y[0]
        plt.text(
            X, Y,
            f"H\n{round(float(pH))}",
            fontsize=LABEL_FONTSIZE, color="blue",
            ha="center", va="center", fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.2")
        )
        nH_draw += 1

print(f"Etichette disegnate: H={nH_draw}  L={nL_draw}")

# =========================
# Output
# =========================
plt.title("SLP (smoothed), clouds, and filtered H/L")
plt.tight_layout()
plt.savefig("out_clean.png", dpi=300)
print(f"Tempo totale: {time.time() - t0:.2f} s")
plt.show()
