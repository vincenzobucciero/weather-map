#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostica percentili su un campo NetCDF (es. SLP):
- stampa pLow/pHigh, conteggi e percentuali dei pixel sotto/sopra soglia
- salva istogramma con linee dei percentili
- salva mappe (maschere) delle aree selezionate
- opzionale: mostra le figure a schermo

Dipendenze minime: numpy, netCDF4, matplotlib
Opzionale per le mappe: basemap (se non presente, fa pcolormesh "semplice")

Esempi:
  python diagnostica_percentili.py wrf5_d02_20250913Z1200.nc --var SLP
  python diagnostica_percentili.py file.nc --var SLP --low 10 --high 90 --show
  python diagnostica_percentili.py file.nc --var SLP --latvar latitude --lonvar longitude --prefix out_diag
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Basemap è opzionale
try:
    from mpl_toolkits.basemap import Basemap
    HAS_BASEMAP = True
except Exception:
    HAS_BASEMAP = False


def read_var_2d(nc, varname):
    """Legge una variabile e la riduce a 2D (gestendo eventuale dimensione time)."""
    var = nc[varname]
    data = var[:]
    # data può essere MaskedArray
    if data.ndim == 3:
        # assume (time, y, x) -> prendi time=0
        data = data[0, :, :]
    elif data.ndim > 3:
        raise ValueError(f"La variabile {varname} ha {data.ndim} dimensioni: gestiscila a mano.")
    return data


def read_coords(nc, latname, lonname):
    """Ritorna (lats, lons) come array 2D."""
    lats = nc[latname][:]
    lons = nc[lonname][:]
    if lats.ndim == 1 and lons.ndim == 1:
        Lons, Lats = np.meshgrid(lons, lats)
        return Lats, Lons
    elif lats.ndim == 2 and lons.ndim == 2:
        return lats, lons
    else:
        raise ValueError("Le coordinate hanno forme incompatibili (devono essere entrambe 1D o entrambe 2D).")


def nanpercentiles(masked_array, perc_low, perc_high):
    """Calcola percentili ignorando masked/NaN in modo sicuro."""
    arr = np.ma.filled(masked_array, np.nan).astype(float)
    p_low, p_high = np.nanpercentile(arr, [perc_low, perc_high])
    valid = np.isfinite(arr)
    return p_low, p_high, arr, valid


def print_stats(arr, valid, p_low, p_high):
    N = np.sum(valid)
    n_low = np.sum(arr[valid] < p_low)
    n_high = np.sum(arr[valid] > p_high)
    print(f"Pixel validi: {N}")
    print(f"pLow={p_low:.2f}   pHigh={p_high:.2f}")
    print(f"low% = {100.0 * n_low / N:.2f}%   high% = {100.0 * n_high / N:.2f}%")
    # sanity check
    p50 = np.nanpercentile(arr[valid], 50)
    if not (p_low < p50 < p_high):
        print("⚠️  Attenzione: p50 non è tra pLow e pHigh. Controlla i dati/maschere.")


def plot_histogram(arr, valid, p_low, p_high, outpath):
    vals = arr[valid].ravel()
    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=60)
    plt.axvline(p_low, color='r', linestyle='--', label=f'pLow={p_low:.2f}')
    plt.axvline(p_high, color='g', linestyle='--', label=f'pHigh={p_high:.2f}')
    plt.title("Distribuzione variabile con percentili")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_mask_map(mask, lats2d, lons2d, title, outpath):
    """Disegna mappa della maschera (1=selezionato, NaN=non selezionato)."""
    data = np.where(mask, 1.0, np.nan)

    if HAS_BASEMAP:
        m = Basemap(
            projection="merc",
            llcrnrlat=float(np.nanmin(lats2d)), urcrnrlat=float(np.nanmax(lats2d)),
            llcrnrlon=float(np.nanmin(lons2d)), urcrnrlon=float(np.nanmax(lons2d)),
            lat_ts=20, resolution="l"
        )
        x, y = m(lons2d, lats2d)
        plt.figure(figsize=(8, 6))
        m.bluemarble(scale=1)
        m.drawcoastlines(linewidth=0.5)
        cs = plt.pcolormesh(x, y, data)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
    else:
        # fallback semplice senza mappa (coordinate come griglia)
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(lons2d, lats2d, data)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(title + " (no basemap)")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Diagnostica percentili su variabili NetCDF 2D (es. SLP).")
    ap.add_argument("ncfile", help="Percorso al file .nc")
    ap.add_argument("--var", default="SLP", help="Nome variabile (default: SLP)")
    ap.add_argument("--latvar", default="latitude", help="Nome variabile latitudine (default: latitude)")
    ap.add_argument("--lonvar", default="longitude", help="Nome variabile longitudine (default: longitude)")
    ap.add_argument("--low", type=float, default=10.0, help="Percentile basso (default: 10)")
    ap.add_argument("--high", type=float, default=90.0, help="Percentile alto (default: 90)")
    ap.add_argument("--prefix", default="diag", help="Prefisso file di output (default: diag)")
    ap.add_argument("--show", action="store_true", help="Mostra anche le figure a schermo")
    args = ap.parse_args()

    nc = Dataset(args.ncfile, "r")
    field = read_var_2d(nc, args.var)
    lats2d, lons2d = read_coords(nc, args.latvar, args.lonvar)

    # percentili nan-safe
    p_low, p_high, arr, valid = nanpercentiles(field, args.low, args.high)
    print_stats(arr, valid, p_low, p_high)

    # istogramma
    plot_histogram(arr, valid, p_low, p_high, f"{args.prefix}_hist.png")
    print(f"Salvato: {args.prefix}_hist.png")

    # mappe maschere
    mask_L = (arr < p_low) & valid
    mask_H = (arr > p_high) & valid
    plot_mask_map(mask_L, lats2d, lons2d, f"Zone L ({args.var} < p{int(args.low)})", f"{args.prefix}_maskL.png")
    plot_mask_map(mask_H, lats2d, lons2d, f"Zone H ({args.var} > p{int(args.high)})", f"{args.prefix}_maskH.png")
    print(f"Salvato: {args.prefix}_maskL.png")
    print(f"Salvato: {args.prefix}_maskH.png")

    if args.show:
        # Riapri e mostra velocemente (opzionale)
        import matplotlib.image as mpimg
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        axs[0].imshow(mpimg.imread(f"{args.prefix}_hist.png")); axs[0].set_title("Istogramma")
        axs[1].imshow(mpimg.imread(f"{args.prefix}_maskL.png")); axs[1].set_title("Mask L")
        axs[2].imshow(mpimg.imread(f"{args.prefix}_maskH.png")); axs[2].set_title("Mask H")
        for ax in axs: ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
