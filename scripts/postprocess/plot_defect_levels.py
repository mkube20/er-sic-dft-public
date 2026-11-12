"""
plot_defect_levels.py: gap-level diagram (Supplementary Fig. 2).

Draws, for each charge state, the levels written by defect_levels.py:
position above the VBM, occupation, dominant orbital character and
localization. Levels are averaged over k points, and the dispersion is shown
as a bar.

Usage:   python scripts/postprocess/plot_defect_levels.py --levels \
             results/defect_levels/er_ko_qp1.json results/defect_levels/er_ko_q0.json \
             results/defect_levels/er_ko_qm1.json --output results/figures/er_ko_defect_levels.pdf
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
import figstyle

CHAR_COLORS = {
    "Er f": "#c0392b",
    "Er d": "#e67e22",
    "O p": "#8e44ad",
    "C p": "#2980b9",
    "Si p": "#7f8c8d",
}
DEFAULT_COLOR = "#95a5a6"


def _color(dominant):
    """Color of an orbital character.
    Used by: _draw_panel, plot.
    Input: dominant channel; returns: color.
    """
    return CHAR_COLORS.get(dominant, DEFAULT_COLOR)


def group_by_band(levels):
    """Average each band's levels over the k points.
    Used by: _draw_panel.
    Input: levels; returns: one entry per spin and band, with the dispersion.
    """
    by_band = defaultdict(list)
    for lv in levels:
        by_band[(lv.get("spin", 0), lv["band_index"])].append(lv)
    out = []
    for (spin, band), group in sorted(by_band.items()):
        e = np.array([g["energy_above_vbm_eV"] for g in group])
        occ = np.array([g["occupation"] for g in group])
        chars = [g.get("orbital_character", {}).get("dominant") for g in group]
        locs = [g.get("localization", {}).get("localization") for g in group]
        locs = [x for x in locs if x is not None]
        dom = max(set(chars), key=chars.count) if chars else None
        stat = {g.get("occ_status") for g in group}
        status = stat.pop() if len(stat) == 1 else "partial"
        out.append({
            "band": band,
            "spin": spin,
            "e_mean": float(e.mean()),
            "e_min": float(e.min()),
            "e_max": float(e.max()),
            "occ_mean": float(occ.mean()),
            "occ_status": status,
            "dominant": dom,
            "localization": float(np.mean(locs)) if locs else None,
            "n_k": len(group),
        })
    return out


def _draw_panel(ax, data, gap_eV, gap_exp, panel_tag):
    """Draw the gap levels of one charge state.
    Used by: plot.
    Inputs: axes, levels JSON, gap (eV), optional experimental gap, panel tag.
    """
    q = data["charge"]
    bands = group_by_band(data["gap_levels"])

    top_edge = max(gap_eV, gap_exp or 0.0)
    ax.axhspan(-0.6, 0.0, color="#34495e", alpha=0.16, zorder=0)
    ax.axhspan(top_edge, top_edge + 0.6, color="#34495e", alpha=0.16, zorder=0)
    ax.axhline(0.0, color="#34495e", lw=1.2)
    if gap_exp and gap_exp > gap_eV:
        ax.axhspan(gap_eV, gap_exp, color="#34495e", alpha=0.07, zorder=0)
        ax.axhline(gap_eV, color="#34495e", lw=1.0, ls=(0, (4, 3)))
        ax.axhline(gap_exp, color="#34495e", lw=1.2)
    else:
        ax.axhline(gap_eV, color="#34495e", lw=1.2)

    last_label_e = -1e9
    min_label_gap = 0.10 * (gap_eV / 2.5)

    for b in bands:
        e, loc, dom = b["e_mean"], b["localization"], b["dominant"]
        col = _color(dom)
        if b["e_max"] - b["e_min"] > 1e-3:
            ax.plot([0.5, 0.5], [b["e_min"], b["e_max"]], color=col,
                    lw=6, alpha=0.18, solid_capstyle="butt", zorder=1)
        ax.plot([0.13, 0.87], [e, e], color=col, lw=2.0, zorder=3,
                solid_capstyle="round")

        status = b.get("occ_status", "partial")
        if status == "occupied":
            ax.plot(0.5, e, "o", ms=6, color=col, zorder=4)
        else:
            ax.plot(0.5, e, "o", ms=6, mfc="white", mec=col, mew=1.2, zorder=4)
            if status == "partial":
                ax.plot(0.5, e, "o", ms=6, color=col, zorder=5,
                        fillstyle="bottom", markeredgewidth=0)

        if loc is not None and loc >= 0.5 and (e - last_label_e) > min_label_gap:
            ax.annotate(f"{loc:.2f}", (0.90, e), fontsize=figstyle.BASE - 1,
                        va="center", ha="left", color=col)
            last_label_e = e

    ax.set_xlim(0, 1.18)
    ax.set_ylim(-0.6, top_edge + 0.6)
    ax.set_xticks([])
    label = f"$q={q:+d}$" if q else "$q=0$"
    ax.text(0.04, 0.97, panel_tag, transform=ax.transAxes, va="top", ha="left",
            fontsize=figstyle.BASE + 1, fontweight="bold")
    ax.text(0.15, 0.97, label, transform=ax.transAxes, va="top", ha="left",
            fontsize=figstyle.BASE + 1)


def plot(level_files, output, gap_eV, gap_exp=None):
    """Draw the gap-level figure.
    Used by: main.
    Inputs: level JSON files, output path, gap (eV), optional experimental gap;
    returns: figure.
    """
    figstyle.apply()
    n = len(level_files)
    ncol = 2 if n > 2 else n
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, sharey=True,
                             figsize=(figstyle.FULL_W, 2.55 * nrow))
    axes = np.array(axes).reshape(-1)

    tags = [f"({chr(ord('a') + i)})" for i in range(n)]
    for i, path in enumerate(level_files):
        _draw_panel(axes[i], json.loads(Path(path).read_text()),
                    gap_eV, gap_exp, tags[i])
    for j in range(n, len(axes)):
        axes[j].axis("off")

    for r in range(nrow):
        axes[r * ncol].set_ylabel("Energy above VBM (eV)")

    present = []
    for path in level_files:
        for lv in json.loads(Path(path).read_text())["gap_levels"]:
            d = lv.get("orbital_character", {}).get("dominant")
            if d and d not in present:
                present.append(d)
    handles = [Line2D([], [], color=_color(d), lw=2.0, label=d) for d in present]
    handles += [
        Line2D([], [], color="k", marker="o", ls="", ms=6, label="occupied"),
        Line2D([], [], color="k", marker="o", ls="", ms=6, mfc="white", mew=1.2,
               label="empty / partial"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.2, handletextpad=0.5)

    fig.tight_layout(rect=(0.02, 0.06, 1, 1))
    figstyle.save(fig, output)
    return fig


def main():
    """Draw the gap-level diagram.
    Used by: command line.
    Writes --output (PDF and PNG).
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", type=Path, required=True,
                    help="defect_levels.py JSON files, in the plotting order")
    ap.add_argument("--gap", type=float, default=2.212,
                    help="PBE supercell gap (eV) setting the axis")
    ap.add_argument("--gap-exp", type=float, default=None,
                    help="experimental gap (eV) to mark, if wanted")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    plot(args.levels, args.output, args.gap, args.gap_exp)


if __name__ == "__main__":
    main()
