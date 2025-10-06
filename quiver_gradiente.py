#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter

# ================== CONFIG ==================
NC_PATH = r"data/opendap/wrf5/d01/archive/2025/09/13/wrf5_d01_20250913Z1200.nc"

# Mostrare nuvolosità come layer trasparente (True/False)
SHOW_CLOUDS = True
# Smoothing SLP per stabilizzare il gradiente
GAUSS_SIGMA = 5
# Livelli isobare
SLP_MIN, SLP_MAX, SLP_STEP = 960, 1100, 2

# ------ Quiver (vettori gradiente) ------
SHOW_QUIVER   = True     # metti False per nascondere le frecce
QUIVER_EVERY  = 8        # subcampionamento (ogni N punti griglia)
QUIVER_SCALE  = 0.8      # scala lunghezza frecce (aumenta per frecce più lunghe)
QUIVER_ALPHA  = 0.9      # trasparenza frecce (0–1)
QUIVER_CMAP   = "viridis"  # colormap per |∇SLP|

# Salvataggio immagine (None per non salvare)
SAVE_PATH = "out_quiver_gradient.png"
DPI = 300
# =========================================

def main():
    # --- Caricamento ---
    f = Dataset(NC_PATH, "r")
    lats = f["latitude"][:]
    lons = f["longitude"][:]
    Lons, Lats = np.meshgrid(lons, lats)

    # Campi
    slp_raw = f["SLP"][0, :, :]
    if SHOW_CLOUDS:
        clf = f["CLDFRA_TOTAL"][0, :, :]
        clf_cmap = LinearSegmentedColormap.from_list(
            "transparent_clouds", [(1, 1, 1, 0), (1, 1, 1, 1)], N=256
        )

    # --- Preprocessing ---
    slp = gaussian_filter(slp_raw, sigma=GAUSS_SIGMA)
    levels = np.arange(SLP_MIN, SLP_MAX + SLP_STEP, SLP_STEP)

    # --- Basemap identica a mainTest.py ---
    m = Basemap(projection="merc",
                llcrnrlat=lats[0], urcrnrlat=lats[-1],
                llcrnrlon=lons[0], urcrnrlon=lons[-1],
                lat_ts=20, resolution="h")
    # m.bluemarble(scale=4)
    m.drawcoastlines()

    x, y = m(Lons, Lats)

    # (Opzionale) nubi semi-trasparenti
    if SHOW_CLOUDS:
        plt.contourf(x, y, clf, cmap=clf_cmap, zorder=1)

    # Isobare
    plt.contour(x, y, slp, colors="yellow", levels=levels,
                linewidths=.5, linestyles="solid", zorder=2)

    # --- Gradiente & modulo ---
    gy, gx = np.gradient(slp)           # dSLP/dlat, dSLP/dlon in unità grid
    grad_mag = np.hypot(gx, gy)

    # --- Quiver opzionale: −∇SLP (verso bassa pressione) ---
    if SHOW_QUIVER:
        step = QUIVER_EVERY
        gx_s = gx[::step, ::step]
        gy_s = gy[::step, ::step]
        x_s  = x[::step, ::step]
        y_s  = y[::step, ::step]

        # Normalizza per mostrare solo direzione; lunghezza uniforme scalata
        mag_s = np.hypot(gx_s, gy_s)
        eps = 1e-12
        ux = -gx_s / (mag_s + eps)  # −∇SLP
        uy = -gy_s / (mag_s + eps)

        # Stima di una lunghezza base in coordinate mappa
        dx_mean = np.nanmean(np.abs(np.diff(x, axis=1)))
        dy_mean = np.nanmean(np.abs(np.diff(y, axis=0)))
        base_len = QUIVER_SCALE * 0.6 * np.nanmin([dx_mean, dy_mean])

        U = ux * base_len
        V = uy * base_len

        # Colora per intensità locale del gradiente (subcampionata)
        color_s = mag_s

        q = plt.quiver(
            x_s, y_s, U, V, color_s,
            angles="xy", scale_units="xy", scale=1.0,
            width=0.0018, headwidth=3.0, headlength=4.5,
            alpha=QUIVER_ALPHA, zorder=6, cmap=QUIVER_CMAP
        )

        cbar = plt.colorbar(q, shrink=0.8, pad=0.01)
        cbar.set_label(r"|$\nabla$SLP| (rel.)")
        plt.quiverkey(q, 0.88, 0.08, base_len, "−∇SLP", labelpos="E", coordinates="axes")

    plt.title("Vettori del gradiente (−∇SLP) con isobare")
    if SAVE_PATH:
        plt.savefig(SAVE_PATH, dpi=DPI, bbox_inches="tight")
        print(f"Salvato: {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
