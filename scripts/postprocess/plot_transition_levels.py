"""
plot_transition_levels.py: formation energy against Fermi level (main-text
Fig. 5 and Supplementary Figs. 3 and 5).

Draws E_f(q; E_F) for each charge state, the lower envelope and the transition
levels, from a JSON written by charge_transition_levels.py.

Usage:   python scripts/postprocess/plot_transition_levels.py \
             --input results/transition_levels/er_ko_transition_levels.json \
             --output results/figures/er_ko_transition_diagram.pdf
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import figstyle

Q_COLORS = {2: "#CC79A7", 1: "#E69F00", 0: "#0072B2", -1: "#009E73", -2: "#56B4E9"}

Q_STYLES = {2: (0, (6, 2)), 1: "solid", 0: (0, (1, 1.8)),
            -1: (0, (6, 1.6, 1, 1.6)), -2: (0, (2.5, 1.6))}


def _as_int(k):
    """Convert a charge key ('1', '+1', '-2' or int) to int.
    Used by: main.
    Input: key; returns: int.
    """
    return int(str(k).replace("+", ""))


def main():
    """Draw the charge-transition diagram.
    Used by: command line.
    Reads --input (JSON); writes --output (PDF and PNG).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="…_transition_levels.json from charge_transition_levels.py --from-abo")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--gap-exp", type=float, default=None,
                    help="Experimental gap (eV) to mark as a second conduction edge. "
                         "OFF BY DEFAULT AND DELIBERATELY UNUSED IN THE MANUSCRIPT "
                         "FIGURES -- see the note below before enabling it.")
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    config = data.get("config", args.input.stem)
    gap = data.get("gap_eV")
    sweep = data.get("sweep")
    if sweep is None:
        raise SystemExit("This JSON has no 'sweep' data — regenerate it with "
                         "charge_transition_levels.py --from-abo (the --input mode "
                         "does not store the sweep).")
    transitions = data.get("transition_levels", [])
    stable = set(data.get("stable_charge_states", []))
    corrected = data.get("corrected", False)

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(figstyle.COL_W, 3.1))

    charges = sorted((_as_int(k) for k in sweep), reverse=True)
    ef_grid = sweep[str(charges[0]) if str(charges[0]) in sweep else charges[0]]["e_fermi"] \
        if isinstance(next(iter(sweep)), str) else sweep[charges[0]]["e_fermi"]

    def series(q):
        """Formation energy of one charge state across the gap.
        Used by: main.
        Input: q; returns: array (eV).
        """
        key = str(q) if str(q) in sweep else (f"+{q}" if f"+{q}" in sweep else q)
        return sweep[key]

    x_end = max(gap or 0.0, args.gap_exp or 0.0) or ef_grid[-1]
    if x_end > ef_grid[-1]:
        step = ef_grid[1] - ef_grid[0]
        n_extra = int(round((x_end - ef_grid[-1]) / step))
        ef_grid = ef_grid + [ef_grid[-1] + step * (i + 1) for i in range(n_extra)]

    envelope = None
    for q in charges:
        s = series(q)
        xs, ys = s["e_fermi"], s["e_f"]
        if len(ef_grid) > len(xs):
            ef0 = ys[0] - q * xs[0]
            xs = ef_grid
            ys = [ef0 + q * x for x in ef_grid]
        c = Q_COLORS.get(q, "0.5")
        lbl = (f"$q={q:+d}$" if q else "$q=0$") + ("" if q in stable else "")
        ax.plot(xs, ys, color=c, lw=(2.0), ls=Q_STYLES.get(q, "solid"),
                alpha=(1.0 if q in stable else 0.6), label=lbl, zorder=3,
                solid_capstyle="round", dash_capstyle="round")
        envelope = ys[:] if envelope is None else [min(a, b) for a, b in zip(envelope, ys)]

    env_line, = ax.plot(ef_grid, envelope, color="0.55", lw=6.0, alpha=0.30,
                        zorder=1, solid_capstyle="round",
                        label="Stable charge-state")

    if gap:
        xmax = max(gap, args.gap_exp or 0.0)
        ax.set_xlim(-0.02 * xmax, xmax * 1.04)
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    y0, y1 = y0 - 0.06 * span, y1 + 0.14 * span
    ax.set_ylim(y0, y1)

    ax.axvline(0.0, color="0.6", lw=0.8)
    if gap:
        ax.axvline(gap, color="0.6", lw=0.8)
        if args.gap_exp:
            ax.axvspan(gap, args.gap_exp, color="0.85", alpha=0.35, zorder=0)
            ax.axvline(args.gap_exp, color="0.35", lw=0.8, ls=(0, (4, 3)))
        y_tag = y1 - 0.025 * (y1 - y0)
        ax.text(0.0, y_tag, "VBM ", va="top", ha="left",
                color="0.4", rotation=90, fontsize=figstyle.BASE - 1)
        ax.text(gap, y_tag, "CBM ", va="top", ha="right",
                color="0.4", rotation=90, fontsize=figstyle.BASE - 1)
        if args.gap_exp:
            ax.text(args.gap_exp, y_tag, "CBM (exp.) ", va="top", ha="right",
                    color="0.35", rotation=90, fontsize=figstyle.BASE - 1)

    eps_label = ('Transition level')
    _y0, _y1 = ax.get_ylim()
    eps_y = min(max(-1.0, _y0 + 0.06 * (_y1 - _y0)), _y1 - 0.06 * (_y1 - _y0))
    for t in transitions:
        eps = t["epsilon_eV"]
        if 0 <= eps <= (gap or ef_grid[-1]):
            k = min(range(len(ef_grid)), key=lambda i: abs(ef_grid[i] - eps))
            ax.axvline(eps, color="0.4", lw=0.8, ls=(0, (1, 2)), zorder=2,
                       label=eps_label)
            q_lo = f"{t['q_lo']:+d}" if t["q_lo"] else "0"
            q_hi = f"{t['q_hi']:+d}" if t["q_hi"] else "0"
            ax.annotate(rf"$\varepsilon({q_lo}/{q_hi})$",
                        xy=(eps, eps_y), xycoords="data",
                        xytext=(0, 0), textcoords="offset points",
                        ha="center", va="bottom", fontsize=figstyle.BASE - 1,
                        color="0.15", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.85))

    ax.set_xlabel(r"Fermi level $E_F$ above VBM (eV)")
    ax.set_ylabel(r"Formation energy $E_f$ (eV)")
    handles, labels = ax.get_legend_handles_labels()
    keep = [n for n, lb in enumerate(labels) if "envelope" not in lb]
    tail = [n for n, lb in enumerate(labels) if "envelope" in lb]
    handles = [handles[n] for n in keep + tail]
    labels = [labels[n] for n in keep + tail]
    ncol = min(len(charges), 2)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.97, bottom=0.28)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=2.0,
               handletextpad=0.7, handlelength=3.2)

    out = args.output or (args.input.parent / f"{config}_transition_diagram.pdf")
    figstyle.save(fig, out)


if __name__ == "__main__":
    main()
