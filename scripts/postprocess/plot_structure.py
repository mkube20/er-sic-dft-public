"""
plot_structure.py: structure files for rendering the relaxed defect.

Writes the relaxed 128-atom cell (CIF, XSF) and, with --view-block 2, a 2x2x1
region around the defect (results/figures/er_ko_view_2x2x1.cif) used for the
VESTA rendering of main-text Fig. 3 (scene file er_ko_view_2x2x1.vesta). Also
draws a matplotlib overview with the relaxed bond lengths.

Usage:   python scripts/postprocess/plot_structure.py --view-block 2
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from defect_levels import _gsr_structure
import figstyle

COLORS = {"Si": "#b9c2cc", "C": "#8d8d8d", "Er": "#c0392b", "O": "#8e44ad"}
RADII = {"Si": 0.42, "C": 0.30, "Er": 0.62, "O": 0.32}
HOST = ("Si", "C")


def _rotation(elev_deg, azim_deg):
    """Rotation matrix for an orthographic view.
    Used by: _project.
    Inputs: elevation and azimuth (degrees); returns: 3x3 matrix.
    """
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return rx @ rz


def _project(coords, elev, azim):
    """Project Cartesian coordinates onto the screen.
    Used by: _draw, main.
    Inputs: coordinates, elevation, azimuth; returns: (x, y, depth).
    """
    r = _rotation(elev, azim)
    p = coords @ r.T
    return p[:, 0], p[:, 2], p[:, 1]


def _draw(ax, coords, symbols, elev, azim, *, scale=1.0, bonds=None,
          highlight=None, labels=None, alpha_host=1.0, label_offset=0.42):
    """Draw atoms and bonds.
    Used by: main.
    Inputs: axes, coordinates, symbols, view angles; returns: screen
    coordinates (x, y).
    """
    x, y, depth = _project(coords, elev, azim)
    order = np.argsort(depth)

    if bonds:
        for i, j in bonds:
            ax.plot([x[i], x[j]], [y[i], y[j]], color="#4a4a4a",
                    lw=1.3, zorder=1, alpha=0.9, solid_capstyle="round")

    for k in order:
        s = symbols[k]
        is_def = highlight is None or k in highlight
        ax.add_patch(plt.Circle(
            (x[k], y[k]), RADII[s] * scale,
            facecolor=COLORS[s], edgecolor="#2b2b2b",
            lw=0.7 if is_def else 0.35,
            alpha=1.0 if is_def else alpha_host,
            zorder=2 + float(depth[k]) / 100.0))

    if labels:
        for (i, j), text in labels.items():
            xm, ym = (x[i] + x[j]) / 2, (y[i] + y[j]) / 2
            dx, dy = x[j] - x[i], y[j] - y[i]
            n = np.hypot(dx, dy) or 1.0
            px, py = -dy / n, dx / n
            ax.annotate(text, (xm + px * label_offset, ym + py * label_offset),
                        fontsize=figstyle.BASE - 1, ha="center", va="center",
                        zorder=50, color="#1a1a1a",
                        bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                  ec="#cccccc", lw=0.4, alpha=0.92))
    ax.set_aspect("equal")
    ax.axis("off")
    return x, y


def _cut_in_gap(vals, target, min_gap=0.02):
    """Move a cut plane into the nearest gap between atomic layers.
    Used by: view_block.
    Inputs: coordinates, target, minimum gap; returns: cut position.
    """
    v = np.sort(np.mod(vals, 1.0))
    edges = np.concatenate([v, [v[0] + 1.0]])
    gaps = np.diff(edges)
    mids = edges[:-1] + gaps / 2.0
    d = np.abs(((mids - target + 0.5) % 1.0) - 0.5)
    d = np.where(gaps > min_gap, d, np.inf)
    if not np.isfinite(d).any():
        raise RuntimeError("no gap wide enough to cut a sub-block cleanly")
    return float(mids[int(np.argmin(d))])


def view_block(st, n, center_index, n_super=4):
    """Cut an n x n x 1 block around the defect from the supercell.
    Used by: main.
    Inputs: structure, n, defect index, supercell size; returns: pymatgen
    Structure.
    """
    from pymatgen.core import Lattice, Structure

    f = st.frac_coords % 1.0
    frac = n / float(n_super)
    cuts = [_cut_in_gap(f[:, ax], f[center_index, ax] - frac / 2) for ax in (0, 1)]
    dx = (f[:, 0] - cuts[0]) % 1.0
    dy = (f[:, 1] - cuts[1]) % 1.0
    sel = (dx < frac) & (dy < frac)
    want = int(round(len(st) * frac * frac))
    if int(sel.sum()) != want:
        raise RuntimeError(f"sub-block selected {sel.sum()} atoms, expected {want}")
    L = st.lattice.matrix.copy()
    L[0] *= frac
    L[1] *= frac
    idx = np.where(sel)[0]
    newf = np.column_stack([dx[sel] / frac, dy[sel] / frac, f[sel, 2]])
    return Structure(Lattice(L), [st[i].specie for i in idx], newf)


def main():
    """Write the structure files and draw the overview figure.
    Used by: command line.
    Writes CIF, XSF and --output.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defect-abo", type=Path,
                    default=Path("abinit/production/er_ko_relax_spin.abo"))
    ap.add_argument("--pristine-abo", type=Path,
                    default=Path("abinit/production/pristine_supercell_gamma.abo"))
    ap.add_argument("--view-block", type=int, default=None, metavar="N",
                    help="ALSO write an NxNx1 sub-block CIF carved from the "
                         "relaxed cell around the defect, for rendering only "
                         "(2 gives the 32-atom block). Named <stem>_view_NxNx1 "
                         "and headed with a warning: it is a crop of the "
                         "128-atom relaxation, not a 32-atom calculation.")
    ap.add_argument("--cutoff", type=float, default=2.6,
                    help="bond cutoff (Ang) for the local-coordination panel")
    ap.add_argument("--output", type=Path,
                    default=Path("results/figures/er_ko_structure.pdf"))
    args = ap.parse_args()

    st = _gsr_structure(args.defect_abo)
    pr = _gsr_structure(args.pristine_abo)

    sym = [s.specie.symbol for s in st]
    er = sym.index("Er")
    ox = sym.index("O")

    si0 = [i for i, s in enumerate(pr) if s.specie.symbol == "Si"][0]
    d_sic = np.mean([n.nn_distance for n in pr.get_neighbors(pr[si0], 2.4)])

    figstyle.apply()
    fig = plt.figure(figsize=(figstyle.FULL_W, 3.05))
    axA = fig.add_axes([0.01, 0.02, 0.44, 0.90])
    axB = fig.add_axes([0.48, 0.02, 0.50, 0.90])

    cart = np.array([s.coords for s in st])
    cart = cart - cart.mean(axis=0)
    _draw(axA, cart, sym, elev=62, azim=15, scale=0.62,
          highlight={er, ox}, alpha_host=0.45)
    figstyle.panel_label(axA, "a")

    L = st.lattice.matrix
    corners = np.array([i * L[0] + j * L[1] + k * L[2]
                        for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    corners = corners - np.array([s.coords for s in st]).mean(axis=0)
    cx, cy, _ = _project(corners, 62, 15)
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for i, j in edges:
        axA.plot([cx[i], cx[j]], [cy[i], cy[j]], color="#9aa4ae", lw=0.7,
                 ls=(0, (3, 2)), zorder=0)

    core = [er, ox]
    shell = set()
    for i in core:
        for n in st.get_neighbors(st[i], args.cutoff):
            shell.add(n.index)
    idx = sorted(set(core) | shell)
    pos = {}
    for i in idx:
        _, img = st.lattice.get_distance_and_image(st[er].frac_coords,
                                                   st[i].frac_coords)
        pos[i] = st.lattice.get_cartesian_coords(st[i].frac_coords + img)
    sub = np.array([pos[i] for i in idx])
    sub = sub - pos[er]
    subsym = [sym[i] for i in idx]
    local = {g: k for k, g in enumerate(idx)}

    bonds, labels = [], {}
    for i in core:
        for n in st.get_neighbors(st[i], args.cutoff):
            if n.index not in local:
                continue
            a, b = local[i], local[n.index]
            if (b, a) in bonds or (a, b) in bonds:
                continue
            bonds.append((a, b))
            labels[(a, b)] = f"{n.nn_distance:.2f}"

    _draw(axB, sub, subsym, elev=8, azim=32, scale=0.80,
          bonds=bonds, labels=labels, label_offset=0.58)
    figstyle.panel_label(axB, "b")
    axB.annotate(f"pristine Si–C bond: {d_sic:.2f} Å",
                 xy=(0.99, 0.02), xycoords="axes fraction",
                 fontsize=figstyle.BASE - 1, ha="right", va="bottom",
                 color="#444444")

    handles = [Line2D([], [], marker="o", ls="", markersize=6,
                      markerfacecolor=COLORS[s], markeredgecolor="#2b2b2b",
                      markeredgewidth=0.5, label=s) for s in ("Er", "O", "Si", "C")]
    axB.legend(handles=handles, fontsize=figstyle.BASE - 1, loc="upper left",
               frameon=False, ncol=4, handletextpad=0.2, columnspacing=0.9)

    figstyle.save(fig, args.output)

    out_dir = args.output.parent
    stem = args.defect_abo.stem.split("_relax")[0].split("_static")[0]
    cif = out_dir / f"{stem}_relaxed.cif"
    st.to(filename=str(cif))
    print(f"Written: {cif}")
    try:
        from pymatgen.io.xcrysden import XSF
        xsf = out_dir / f"{stem}_relaxed.xsf"
        xsf.write_text(XSF(st).to_str())
        print(f"Written: {xsf}")
    except Exception as exc:
        print(f"  (XSF export skipped: {exc})")

    if args.view_block:
        n = args.view_block
        sub = view_block(st, n, er)
        vcif = out_dir / f"{stem}_view_{n}x{n}x1.cif"
        sub.to(filename=str(vcif))
        banner = (
            f"# DISPLAY ONLY -- {n}x{n}x1 crop of the relaxed 4x4x1 (128-atom)"
            f" cell.\n"
            f"# Coordinates are from the converged 128-atom calculation; the"
            f" defect complex\n"
            f"# is exact. The CELL is not: atoms on the new faces were relaxed"
            f" against\n"
            f"# 128-atom neighbors and now face different images, so contacts"
            f" across those\n"
            f"# faces are artifacts. This is NOT a {len(sub)}-atom calculation"
            f" of this defect,\n"
            f"# and the {len(sub)}-atom cell used elsewhere in this project is"
            f" the superseded,\n"
            f"# unpolarized screening cell. Render from this file; quote from"
            f" {stem}_relaxed.cif.\n"
        )
        vcif.write_text(banner + vcif.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Written: {vcif}   ({len(sub)} atoms, display only)")

    print("\nGeometry for the caption:")
    print(f"  Er-O                 {st.get_distance(er, ox):.3f} Ang")
    for n in sorted(st.get_neighbors(st[er], args.cutoff),
                    key=lambda n: n.nn_distance):
        print(f"  Er-{n.specie.symbol:<2s}                {n.nn_distance:.3f} Ang")
    for n in sorted(st.get_neighbors(st[ox], args.cutoff),
                    key=lambda n: n.nn_distance):
        if n.specie.symbol != "Er":
            print(f"  O-{n.specie.symbol:<2s}                 {n.nn_distance:.3f} Ang")
    print(f"  pristine Si-C        {d_sic:.3f} Ang")


if __name__ == "__main__":
    main()
