"""
plot_production_formation_energies.py: formation energies of the nineteen
configurations (main-text Fig. 2), including the oxygen and nitrogen complexes
at the SiO2 and Si3N4 solubility limits.

Input:   results/formation_energies/formation_energies_spin.json,
         results/formation_energies/solubility_limits.json
Output:  results/figures/formation_energy_production.pdf
Usage:   python scripts/postprocess/plot_production_formation_energies.py
"""

from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import figstyle

REPO = Path(__file__).resolve().parents[2]
_SPIN_JSON = REPO / "results/formation_energies/formation_energies_spin.json"
PROD_JSON = (_SPIN_JSON if _SPIN_JSON.exists()
             else REPO / "results/formation_energies/formation_energies_production.json")
SCREEN_JSON = REPO / "results/formation_energies/formation_energies.json"
BOUND_JSON = REPO / "results/formation_energies/solubility_limits.json"
OUT = REPO / "results/figures/formation_energy_production.pdf"
OUT_PNG = REPO / "results/figures/formation_energy_production.png"

GROUPS = [
    ("er_ho",  "er_ko",  r"Er$_{\rm Si}$+O$_{\rm C}$"),
    ("er_hn",  "er_kn",  r"Er$_{\rm Si}$+N$_{\rm C}$"),
    ("er_hp",  "er_kp",  r"Er$_{\rm Si}$+P$_{\rm Si}$"),
    ("er_hv",  "er_kv",  r"Er$_{\rm Si}$+V$_{\rm C}$"),
    ("er_h",   "er_k",   r"Er$_{\rm Si}$"),
    ("er_hal", "er_kal", r"Er$_{\rm Si}$+Al$_{\rm Si}$"),
    ("er_hb",  "er_kb",  r"Er$_{\rm Si}$+B$_{\rm Si}$"),
    ("er_hc",  "er_kc",  r"Er$_{\rm C}$"),
    ("er_i",   None,     r"Er$_{\rm i}$"),
    ("er_hh", "er_kh", r"Er$_{\rm Si}$+H$_{\rm i}$"),
]

C_H = "#0072B2"
C_K = "#E69F00"
C_NONE = "#7A7A7A"
C_PEND = "#9AA0A6"

HERO_EDGE_LW = 1.3
BASE_EDGE_LW = 0.5


def main():
    """Draw the formation-energy figure.
    Used by: command line.
    Writes results/figures/formation_energy_production.pdf.
    """
    data = json.loads(PROD_JSON.read_text())
    fe = data["formation_energies_eV"]
    hero = data.get("lowest", data.get("hero"))

    def val(c):
        """Si-rich, C-rich and mean formation energies of a configuration.
        Used by: main.
        Input: config; returns: (min, max, mean) in eV, or None.
        """
        if c is None or c not in fe:
            return None
        a, b = fe[c]["Si_rich"], fe[c]["C_rich"]
        return min(a, b), max(a, b), 0.5 * (a + b)


    def group_key(g):
        """Sort key of a defect group: its lowest mean formation energy.
        Used by: main.
        Input: group; returns: eV.
        """
        vals = [val(c) for c in g[:2]]
        got = [v[2] for v in vals if v is not None]
        return min(got) if got else float("inf")

    groups = sorted(GROUPS, key=group_key)

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(figstyle.FULL_W, 3.4))

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", lw=0.6, zorder=0)
    ax.xaxis.grid(False)

    W = 0.42
    ymax = 0.0
    drawn_sites, hero_drawn, any_lim, any_pending = set(), False, False, False
    any_bound = False
    bounds = (json.loads(BOUND_JSON.read_text())["formation_energy_at_bound_eV"]
              if BOUND_JSON.exists() else {})

    for gi, (c_h, c_k, _label) in enumerate(groups):
        if c_h is not None and c_k is not None:
            members = [(c_k, "k"), (c_h, "h")]
        elif c_k is not None:
            members = [(c_k, "k")]
        else:
            members = [(c_h, None)]
        for c, site in members:
            if len(members) == 1:
                xpos = gi
            else:
                xpos = gi + (-W / 2 if site == "k" else W / 2)

            v = val(c)
            if v is None:
                any_pending = True
                ax.annotate("pending", (xpos, 0), textcoords="offset points",
                            xytext=(0, 3), ha="center", va="bottom",
                            rotation=90, color=C_PEND,
                            fontsize=figstyle.BASE - 2.5, zorder=4)
                continue

            v_lo, v_hi, v_mid = v
            color = C_H if site == "h" else C_K if site == "k" else C_NONE
            drawn_sites.add(site)
            is_hero = (c == hero)
            hero_drawn |= is_hero
            ax.bar(xpos, v_mid, W, color=color, edgecolor="black",
                   linewidth=HERO_EDGE_LW if is_hero else BASE_EDGE_LW,
                   zorder=3)

            if v_hi - v_lo > 1e-3:
                any_lim = True
                ax.vlines(xpos, v_lo, v_hi, color="#1A1A1A", lw=0.9, zorder=6)
                ax.scatter(xpos, fe[c]["Si_rich"], s=15, marker="o",
                           facecolor="#1A1A1A", edgecolor="#1A1A1A", lw=0.9,
                           zorder=7)
                ax.scatter(xpos, fe[c]["C_rich"], s=15, marker="o",
                           facecolor="white", edgecolor="#1A1A1A", lw=0.9,
                           zorder=7)

            ax.annotate(f"{v_mid:.2f}", (xpos, v_hi), textcoords="offset points",
                        xytext=(0, 3), ha="center", va="bottom",
                        fontsize=figstyle.BASE - 2, zorder=7)
            ymax = max(ymax, v_hi)

            if c in bounds:
                any_bound = True
                b_si, b_c = bounds[c]["Si_rich"], bounds[c]["C_rich"]
                ax.vlines(xpos, v_hi, max(b_si, b_c), color="#555555",
                          lw=0.8, ls=(0, (1.2, 1.4)), zorder=5)
                ax.scatter(xpos, b_si, s=17, marker="D", facecolor="#1A1A1A",
                           edgecolor="#1A1A1A", lw=0.8, zorder=7)
                ax.scatter(xpos, b_c, s=17, marker="D", facecolor="white",
                           edgecolor="#1A1A1A", lw=0.8, zorder=7)
                ymax = max(ymax, b_si, b_c)

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([g[2] for g in groups], rotation=30, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel(r"Formation energy $E_f$ (eV)")
    ax.set_xlabel("Defect type", labelpad=4)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, ymax * 1.16)
    ax.tick_params(axis="x", length=0)

    handles, labels = [], []
    if "h" in drawn_sites:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=C_H, ec="black", lw=0.5))
        labels.append(r"Er on $h$ site")
    if "k" in drawn_sites:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=C_K, ec="black", lw=0.5))
        labels.append(r"Er on $k$ site")
    if hero_drawn:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc="none", ec="black",
                                     lw=HERO_EDGE_LW))
        labels.append(r"Lowest at O$_2$/N$_2$ reference")
    if any_lim:
        handles += [
            plt.Line2D([0], [0], color="#1A1A1A", lw=0, marker="o", ms=4.0,
                       mfc="#1A1A1A", mec="#1A1A1A", mew=0.9),
            plt.Line2D([0], [0], color="#1A1A1A", lw=0, marker="o", ms=4.0,
                       mfc="white", mec="#1A1A1A", mew=0.9)]
        labels += ["Si-rich limit", "C-rich limit"]
    if any_bound:
        handles.append(plt.Line2D([0], [0], color="#1A1A1A", lw=0, marker="D",
                                  ms=3.6, mfc="#1A1A1A", mec="#1A1A1A",
                                  mew=0.8))
        labels.append(r"At SiO$_2$ / Si$_3$N$_4$ solubility limit")
    leg = ax.legend(handles, labels, frameon=False, loc="upper left",
                    fontsize=figstyle.BASE - 1, handlelength=1.3,
                    borderaxespad=0.4, labelspacing=0.3, ncol=2,
                    columnspacing=1.4)
    _ = any_pending

    fig.tight_layout()
    figstyle.save(fig, OUT)


if __name__ == "__main__":
    main()
