"""
plot_pdos.py: spin-resolved projected DOS of a defect cell (Supplementary Fig. 1).

Panels: Er 4f, Er 5d, C p and Si p of host atoms far from the defect, and the
total over all PAW spheres, relative to the VBM.

Usage:   python scripts/postprocess/plot_pdos.py --run abinit/er_ko_occprobe_spin_run \
             --stem er_ko_occprobe_spino_DS1_DOS_AT --er 75 --output results/figures/er_ko_pdos.pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import figstyle
from pdos_io import read_spin_dos as _dos, gsr_band_edges as _gsr

RUN = Path("abinit/er_ko_occprobe_spin_run")
STEM = "er_ko_occprobe_spino_DS1_DOS_AT"
ER_ATOM = 75
SI_ATOM = 1
C_ATOM = 17

L = {"s": 0, "p": 1, "d": 2, "f": 3}


def series(atom, l):
    """Projected DOS of one atom and channel, per spin.
    Used by: main.
    Inputs: atom index, l; returns: (E up, DOS up, E down, DOS down).
    """
    blocks = _dos(RUN / f"{STEM}{atom:04d}", l=L[l])
    if len(blocks) < 2:
        raise RuntimeError(f"atom {atom}: expected two spin blocks, got {len(blocks)}")
    (e_up, d_up), (e_dn, d_dn) = blocks[0], blocks[1]
    return e_up, d_up, e_dn, d_dn


OCC_ALPHA = 0.85
EMPTY_ALPHA = 0.16


def fill_split(ax, x, y, ef_x, color, label=None):
    """Fill a DOS curve, solid below the Fermi level and pale above.
    Used by: main.
    Inputs: axes, energies, DOS, Fermi level, color, label.
    """
    occ = x <= ef_x
    ax.fill_between(x, 0, y, where=occ, interpolate=True, color=color,
                    alpha=OCC_ALPHA, lw=0, label=label, zorder=3)
    ax.fill_between(x, 0, y, where=~occ, interpolate=True, color=color,
                    alpha=EMPTY_ALPHA, lw=0, zorder=3)
    ax.plot(np.where(occ, np.nan, x), np.where(occ, np.nan, y),
            color=color, lw=0.55, alpha=0.75, zorder=4)

def main():
    """Draw the projected-DOS figure.
    Used by: command line.
    Writes --output (PDF and PNG).
    """
    global RUN, STEM, ER_ATOM
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=Path("results/figures/er_ko_pdos.pdf"))
    ap.add_argument("--run", type=Path, default=None,
                    help="run directory holding the per-atom DOS files "
                         f"(default {RUN})")
    ap.add_argument("--stem", default=None,
                    help="DOS filename stem inside --run "
                         f"(default {STEM})")
    ap.add_argument("--er", type=int, default=None,
                    help=f"1-based index of the Er atom (default {ER_ATOM})")
    ap.add_argument("--e-vbm", type=float, default=7.6953)
    ap.add_argument("--e-cbm", type=float, default=9.9072)
    ap.add_argument("--e-fermi", type=float, default=None,
                    help="Fermi level in eV on the run's own scale, used "
                         "only to split occupied from empty. Default: read "
                         "from the run's _GSR.nc.")
    args = ap.parse_args()
    if args.run is not None:
        RUN = args.run
    if args.stem is not None:
        STEM = args.stem
    if args.er is not None:
        ER_ATOM = args.er

    ef = args.e_vbm
    gap = args.e_cbm - args.e_vbm

    if args.e_fermi is not None:
        ef_abs = args.e_fermi
    else:
        import h5py
        gsr = sorted(RUN.glob("*_GSR.nc"))
        if not gsr:
            raise SystemExit(f"no _GSR.nc in {RUN}; pass --e-fermi")
        with h5py.File(gsr[0], "r") as h:
            ef_abs = float(np.array(h["fermi_energy"])) * 27.211386245988
    ef_x = ef_abs - ef

    figstyle.apply()
    fig, axes = plt.subplots(4, 1, figsize=(figstyle.COL_W * 1.6, 7.2),
                             sharex=True, gridspec_kw={"hspace": 0.12})

    panels = [
        ("Er 4f", [(ER_ATOM, "f", "#C1272D", "Er 4f")]),
        ("Er 5d", [(ER_ATOM, "d", "#0072B2", "Er 5d")]),
        ("host p", [(C_ATOM, "p", "#333333", "C p"),
                    (SI_ATOM, "p", "#7FB3D5", "Si p")]),
    ]

    for ax, (label, entries) in zip(axes[:3], panels):
        for atom, l, color, leg in entries:
            e_up, d_up, e_dn, d_dn = series(atom, l)
            fill_split(ax, e_up - ef, d_up, ef_x, color, label=leg)
            fill_split(ax, e_dn - ef, -d_dn, ef_x, color)
        ax.axhline(0, color="k", lw=0.6, zorder=5)
        ax.set_ylabel(f"{label}\n(states eV$^{{-1}}$)")
        ax.legend(loc="upper left", frameon=False, handlelength=1.2)

    files = sorted(RUN.glob(f"{STEM}*"))
    tot_up = tot_dn = None
    for f in files:
        for l in ("s", "p", "d", "f"):
            blocks = _dos(f, l=L[l])
            if len(blocks) < 2:
                continue
            (e_up, d_up), (e_dn, d_dn) = blocks[0], blocks[1]
            tot_up = d_up if tot_up is None else tot_up + d_up
            tot_dn = d_dn if tot_dn is None else tot_dn + d_dn
    ax = axes[3]
    fill_split(ax, e_up - ef, tot_up, ef_x, "#555555",
               label="total (all spheres)")
    fill_split(ax, e_dn - ef, -tot_dn, ef_x, "#555555")
    ax.axhline(0, color="k", lw=0.6, zorder=5)
    ax.set_ylabel("total\n(states eV$^{-1}$)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlabel("$E - E_{\\rm VBM}$  (eV)")

    for ax in axes:
        ax.axvspan(0, gap, color="#000000", alpha=0.055, lw=0, zorder=0)
        for x in (0.0, gap):
            ax.axvline(x, color="0.45", lw=0.7, zorder=1)
        ax.axvline(ef_x, color="0.25", lw=0.9, ls=(0, (4, 2)), zorder=2)
        ax.set_xlim(-11, 5)

    top = axes[0]
    top.annotate("band gap", xy=(gap / 2, 0.55), xycoords=("data", "axes fraction"),
                 ha="center", va="center", rotation=90,
                 fontsize=figstyle.BASE - 1.5, color="0.45")
    top.annotate("$E_{\\rm F}$", xy=(ef_x, 1.02),
                 xycoords=("data", "axes fraction"), ha="center", va="bottom",
                 fontsize=figstyle.BASE - 1.5, color="0.25")

    from matplotlib.patches import Patch
    key = [Patch(facecolor="0.35", alpha=OCC_ALPHA, label="occupied"),
           Patch(facecolor="0.35", alpha=EMPTY_ALPHA, edgecolor="0.45",
                 linewidth=0.55, label="empty")]
    axes[1].legend(handles=key, loc="upper right", frameon=False,
                   handlelength=1.2, fontsize=figstyle.BASE - 1)

    figstyle.save(fig, args.output)
    print(f"Written: {args.output}   (VBM {ef:.3f} eV, gap {gap:.3f} eV, E_F {ef_x:+.3f} eV above VBM)")


if __name__ == "__main__":
    main()
