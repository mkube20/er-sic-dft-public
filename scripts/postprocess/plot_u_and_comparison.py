"""
plot_u_and_comparison.py: two figures.

u_scan.pdf (Supplementary Fig. 4): eps(+1/0) and the cell moment of
Er_Si(k)+O_C against U, from results/transition_levels/er_ko_u_scan.json.

er_k_vs_er_ko_pdos.pdf (main-text Fig. 4): Er 4f and 5d projected DOS of
isolated Er_Si(k) and of Er_Si(k)+O_C, from the static calculations seeded with
each relaxation's 4f occupation matrix.

Usage:   python scripts/postprocess/plot_u_and_comparison.py --left er_k --right er_ko
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import figstyle
from pdos_io import read_spin_dos as _dos

HA = 27.211386245988

RUNS = {
    "er_h": ("abinit/er_h_static_spin_run", "er_h_static_spino_DS1_DOS_AT0102",
             "abinit/er_h_static_spin_run/er_h_static_spino_DS1_GSR.nc",
             r"Er$_{\rm Si}$(h)"),
    "er_ko": ("abinit/er_ko_occprobe_spin_run", "er_ko_occprobe_spino_DS1_DOS_AT0075",
              "abinit/er_ko_occprobe_spin_run/er_ko_occprobe_spino_DS1_GSR.nc",
              r"Er$_{\rm Si}$(k)+O$_{\rm C}$"),
    "er_k": ("abinit/er_k_occprobe_spin_run", "er_k_occprobe_spino_DS1_DOS_AT0075",
            "abinit/er_k_occprobe_spin_run/er_k_occprobe_spino_DS1_GSR.nc",
            r"Er$_{\rm Si}$(k)"),
}

C_4F = "#C1272D"
C_5D = "#0072B2"
C_ACC = "#B0175A"


def fermi(gsr):
    """Fermi energy of a run.
    Used by: figure_comparison.
    Input: _GSR.nc path; returns: eV.
    """
    with h5py.File(gsr, "r") as h:
        return float(np.array(h["fermi_energy"])) * HA


def figure_u_scan(out):
    """Draw eps(+1/0) and the moment against U.
    Used by: main.
    Input: output path; returns: spread of eps (eV).
    """
    data = json.loads(Path("results/transition_levels/er_ko_u_scan.json").read_text())
    pts = [p for p in data["points"] if "missing" not in p]
    pts.sort(key=lambda p: p["U_eV"])
    u = np.array([p["U_eV"] for p in pts])
    eps = np.array([p["epsilon_p1_0_eV"] for p in pts])
    mom = np.array([p["moment_muB"] for p in pts])
    prod = np.array([p["production"] for p in pts])

    figstyle.apply()
    fig, axes = plt.subplots(2, 1, figsize=(figstyle.COL_W, 4.2), sharex=True,
                             gridspec_kw={"hspace": 0.12, "height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(u, eps, "-", color=C_ACC, lw=1.4, zorder=2)
    ax.scatter(u[~prod], eps[~prod], s=34, color=C_ACC, zorder=3,
               label="U-scan statics")
    ax.scatter(u[prod], eps[prod], s=80, facecolor="white", edgecolor=C_ACC,
               lw=1.8, zorder=4, label="production value")
    spread = eps.max() - eps.min()
    ax.axhspan(eps.min(), eps.max(), color=C_ACC, alpha=0.10, zorder=1)
    ax.set_ylabel(r"$\varepsilon(+1/0)$  (eV above VBM)")
    ax.set_ylim(eps.mean() - 0.42, eps.mean() + 0.42)
    ax.annotate(f"total spread {spread:.3f} eV\nover U = {u.min():g}–{u.max():g} eV",
                xy=(0.03, 0.08), xycoords="axes fraction", fontsize=8.5,
                va="bottom", color=C_ACC)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(u, mom, "-", color=C_5D, lw=1.4, zorder=2)
    ax.scatter(u, mom, s=28, color=C_5D, zorder=3)
    ax.axhspan(2.5, 3.6, color="0.5", alpha=0.13, zorder=1)
    ax.axhline(3.0, color="0.35", lw=0.8, ls="--", zorder=2)
    ax.set_ylabel(r"Er moment ($\mu_{\rm B}$)")
    ax.set_xlabel(r"Hubbard $U$ on Er 4f  (eV)")
    ax.set_ylim(2.45, 3.35)
    ax.annotate(r"4f$^{11}$, $S=3/2$", xy=(0.03, 0.72), xycoords="axes fraction",
                fontsize=8, color="0.35")

    figstyle.save(fig, out)
    print(f"Written: {out}   (spread {spread:.3f} eV)")
    return spread


def broaden(e, d, fwhm):
    """Gaussian broadening of a DOS.
    Used by: figure_comparison.
    Inputs: energies, DOS, FWHM (eV); returns: broadened DOS.
    """
    if not fwhm:
        return d
    de = e[1] - e[0]
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    n = max(1, int(round(4.0 * sigma / de)))
    x = np.arange(-n, n + 1) * de
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return np.convolve(d, k, mode="same")

E_VBM, E_CBM = 7.6953, 9.9072

OCC_ALPHA = 0.85
EMPTY_ALPHA = 0.16


def fill_split(ax, x, y, color):
    """Fill a DOS curve, solid below the Fermi level and pale above.
    Used by: figure_comparison.
    Inputs: axes, energies, DOS, color.
    """
    occ = x <= 0.0
    ax.fill_between(x, 0, y, where=occ, interpolate=True, color=color,
                    alpha=OCC_ALPHA, lw=0, zorder=3)
    ax.fill_between(x, 0, y, where=~occ, interpolate=True, color=color,
                    alpha=EMPTY_ALPHA, lw=0, zorder=3)
    ax.plot(np.where(occ, np.nan, x), np.where(occ, np.nan, y),
            color=color, lw=0.55, alpha=0.75, zorder=4)


def figure_comparison(out, left="er_h", right="er_ko", fwhm=0.0):
    """Draw the Er 4f and 5d projected DOS of two cells side by side.
    Used by: main.
    Inputs: output path, left and right configs, FWHM (eV).
    """
    figstyle.apply()
    fig, axes = plt.subplots(2, 2, figsize=(figstyle.COL_W * 1.9, 4.6),
                             sharex=True, sharey="row",
                             gridspec_kw={"hspace": 0.14, "wspace": 0.07})

    for col, key in enumerate((left, right)):
        d, stem, gsr, title = RUNS[key]
        ef = fermi(gsr)
        for row, (l, color, name) in enumerate(((3, C_4F, "Er 4f"),
                                                (2, C_5D, "Er 5d"))):
            ax = axes[row, col]
            blocks = _dos(Path(d) / stem, l=l)
            (e_up, raw_up), (e_dn, raw_dn) = blocks[0], blocks[1]
            d_up = broaden(e_up, raw_up, fwhm)
            d_dn = broaden(e_dn, raw_dn, fwhm)
            fill_split(ax, e_up - ef, d_up, color)
            fill_split(ax, e_dn - ef, -d_dn, color)
            ax.axhline(0, color="k", lw=0.6)
            ax.axvspan(E_VBM - ef, E_CBM - ef, color="#000000", alpha=0.055,
                       lw=0, zorder=0)
            for xb in (E_VBM - ef, E_CBM - ef):
                ax.axvline(xb, color="0.45", lw=0.7, zorder=1)
            ax.axvline(0, color="0.25", lw=0.9, ls=(0, (4, 2)), zorder=2)
            ax.set_xlim(-9, 5)
            if col == 0:
                ax.set_ylabel(f"{name}\n(states eV$^{{-1}}$)")
            if row == 0:
                ax.set_title(title, fontsize=10, pad=6)

            tot = raw_up + raw_dn
            w = (e_up > E_VBM) & (e_up < E_CBM)
            frac = 100 * np.trapezoid(tot[w], e_up[w]) / np.trapezoid(tot, e_up)
            y, va = (0.05, "bottom") if row == 0 else (0.88, "top")
            ax.annotate(f"{frac:.2f}% in bandgap",
                        xy=(0.035, y), xycoords="axes fraction", fontsize=8,
                        fontweight="bold", va=va, color=color)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    def _pair(alpha, edge=None):
        """Legend patches for occupied and empty states.
        Used by: figure_comparison.
        Inputs: opacity, edge color; returns: tuple of patches.
        """
        return tuple(Patch(facecolor=c, alpha=alpha, edgecolor=edge,
                           linewidth=0.55 if edge else 0.0)
                     for c in (C_4F, C_5D))

    handles = [_pair(OCC_ALPHA), _pair(EMPTY_ALPHA, edge='0.45'),
               Patch(facecolor='#000000', alpha=0.14, edgecolor='0.45',
                     linewidth=0.7),
               Line2D([], [], color='0.25', lw=0.9, ls=(0, (4, 2)))]
    labels = ['Occupied', 'Empty', 'Bandgap', 'Fermi level']
    axes[0, 1].legend(handles, labels, loc='upper right', frameon=False,
                      handlelength=2.2, fontsize=figstyle.BASE - 2,
                      labelspacing=0.35, borderaxespad=0.4,
                      handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})


    for ax in axes[1]:
        ax.set_xlabel(r"$E - E_{\rm F}$  (eV)")
    figstyle.save(fig, out)
    print(f"Written: {out}")


def main():
    """Draw the U-scan and PDOS-comparison figures.
    Used by: command line.
    Writes u_scan.pdf and <left>_vs_<right>_pdos.pdf.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("results/figures"))
    ap.add_argument("--left", default="er_k", choices=sorted(RUNS),
                    help="isolated-defect panel (default er_h; er_k is the "
                         "site-matched control and needs er_k_pot_q0_spin)")
    ap.add_argument("--right", default="er_ko", choices=sorted(RUNS),
                    help="compensated-complex panel")
    ap.add_argument("--fwhm", type=float, default=0.0, metavar="EV",
                    help="Gaussian broadening of the plotted DOS in eV "
                         "(default 0, i.e. none). Quoted percentages are "
                         "measured on the raw DOS regardless.")
    args = ap.parse_args()
    missing = [k for k in (args.left, args.right)
               if not Path(RUNS[k][2]).exists()]
    if missing:
        raise SystemExit(
            f"no projected-DOS run for {', '.join(missing)}: expected "
            f"{', '.join(RUNS[k][2] for k in missing)}. Submit the matching "
            "abinit/production/<config>_pot_q0_spin.abi first.")
    for ext in ("pdf", "png"):
        figure_u_scan(args.outdir / f"u_scan.{ext}")
        figure_comparison(args.outdir / f"{args.left}_vs_{args.right}_pdos.{ext}",
                          left=args.left, right=args.right, fwhm=args.fwhm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
